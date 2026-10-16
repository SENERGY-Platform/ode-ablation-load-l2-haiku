"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

Three methods are yours. Everything else — Kafka, the model registry, the
lifecycle — belongs to the library.

The forecast. Each message, seen at time t, emits the mean net grid power for the
hour that starts 24 h after the start of t's hour, timestamped with that hour. The
evaluation buckets predictions by that timestamp and scores them against the hourly
mean of the target, so every message is a forecast for one hour and the hour gets
an average over the messages that forecast it.

The level offset. The profile is the same for every day of a weekday or weekend
slot, so it misses the day's level. The forecast adds the mean, over the previous
seven days at the forecast hour's own hour of day, of observed hourly mean minus
profile value. Only complete hours are used: the hours are 24, 48, ... 168 hours
before the current hour, so nothing in the forecast reads the hour it is about. The
offset is recomputed once per hour, not per message.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import HORIZON, forecast_hour, train_model

_HOUR = datetime.timedelta(hours=1)
_UTC = datetime.timezone.utc

# Same-hour-of-day lookback for the offset: seven days.
_OFFSET_DAYS = 7
# The buffer must reach back past the oldest hour the offset reads: 168 h plus
# a margin, so a pruned hour is never one the offset still needs.
_BUFFER_HOURS = 24 * _OFFSET_DAYS + 2


def _hour_start(at: datetime.datetime) -> datetime.datetime:
    """The start of the UTC hour `at` falls into."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=_UTC)
    at = at.astimezone(_UTC)
    return at.replace(minute=0, second=0, microsecond=0)


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


class Operator(MLOperator):
    configType = CustomConfig

    # Which inputs this operator accepts. "args" are the mapping destinations the
    # pipeline is configured with, and the name is what infer() receives as
    # "selector", so one operator can treat several input shapes differently.
    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        # Hour start -> [sum, count] of observed values in that hour.
        self._hourly: typing.Dict[datetime.datetime, list] = {}
        self._offset_hour: typing.Optional[datetime.datetime] = None
        self._offset = 0.0
        super().init(*args, **kwargs)

    def _level_offset(self, model: PyFuncModel, hour: datetime.datetime) -> float:
        """Mean of observed minus profile over the same hour of day, previous seven days.

        `hour` is the current hour; the forecast is for hour + 24 h. The same hour of
        day as the forecast, on each of the seven previous days, is hour - 24k h for
        k = 1..7, all complete. The profile value for a past hour h is the model's
        forecast made 24 h before h, so the offset uses the model's own contract.
        Zero when none of those hours has data.
        """
        diffs = []
        for k in range(1, _OFFSET_DAYS + 1):
            past = hour - k * 24 * _HOUR
            acc = self._hourly.get(past)
            if acc is None or acc[1] == 0:
                continue
            observed = acc[0] / acc[1]
            profile = float(model.predict({"timestamp": (past - HORIZON).isoformat()}))
            diffs.append(observed - profile)
        return sum(diffs) / len(diffs) if diffs else 0.0

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        The result timestamp is the hour the forecast is about, 24 h after the
        message's hour, so the evaluation and any consumer read it at the time it
        describes.
        """
        if model is None:
            return None, None, None

        hour = _hour_start(timestamp)

        value = data.get("value")
        if value is not None:
            acc = self._hourly.setdefault(hour, [0.0, 0])
            acc[0] += float(value)
            acc[1] += 1

        if self._offset_hour != hour:
            self._offset = self._level_offset(model, hour)
            self._offset_hour = hour
            # Hours older than the lookback are no longer needed.
            cutoff = hour - _BUFFER_HOURS * _HOUR
            for old in [h for h in self._hourly if h < cutoff]:
                del self._hourly[old]

        prediction = model.predict({"timestamp": timestamp.isoformat()}) + self._offset
        return forecast_hour(timestamp), {"prediction": float(prediction)}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started — so params and metrics logged through "logger" land on the run
        the resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Time-based to start with. A better policy watches the data — a drift in the
        input distribution, or an error that stops falling — and that is a decision
        to make with the profile in front of you rather than a default to inherit.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
