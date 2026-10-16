"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

The model is a profile of net grid power: the mean hourly power for each
combination of weekend flag and local hour of day, fitted on the training window.
The forecast for the hour H is that profile value for H, produced by a message
seen 24 hours before H, so the prediction carries the time it is about.

provide_historic_data() returns one Ray ObjectRef per input topic, each to a Ray
Dataset. Those are resolved here in the driver, which is where train() runs.
"""

import datetime
import typing

import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads. Ends at the clock: training_end under
# a data split, now otherwise. 28 days, not 90: the 90-day window reached back into
# June, and a summer profile applied to September was the season-drift hypothesis
# the baseline was meant to test.
TRAINING_WINDOW = datetime.timedelta(days=28)

# The site's local clock. The profile is by local hour, so a daily load shape
# lands on the same hour in summer and winter.
LOCAL_TZ = "Europe/Berlin"

# How far ahead each forecast looks.
HORIZON = datetime.timedelta(hours=24)

_UTC = datetime.timezone.utc


def forecast_hour(seen_at: datetime.datetime) -> datetime.datetime:
    """The hour a message seen at `seen_at` forecasts: the start of the hour
    HORIZON after it. This is the result timestamp the evaluation buckets by."""
    if seen_at.tzinfo is None:
        seen_at = seen_at.replace(tzinfo=_UTC)
    seen_at = seen_at.astimezone(_UTC)
    return seen_at.replace(minute=0, second=0, microsecond=0) + HORIZON


def _slot(local: pd.DatetimeIndex) -> pd.Index:
    """Profile slot 0..47: weekday hours first, weekend hours after."""
    weekend = (local.dayofweek >= 5).astype(int)
    return weekend * 24 + local.hour


class OdeAblationLoadL2HaikuModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives exactly the payload infer() builds, so the two are one
    contract in two files. Keep it serialisable: MLflow stores this object.
    """

    def __init__(self, profile: typing.List[float]) -> None:
        # 48 values: weekday hour 0..23, then weekend hour 0..23.
        self.profile = profile

    def predict(self, context, model_input=None, params=None):
        # The pyfunc signature carries a context when MLflow calls it and not when the
        # model is called directly, so the payload is taken from whichever argument
        # holds it.
        payload = model_input if model_input is not None else context
        seen_at = payload["timestamp"]
        if isinstance(seen_at, str):
            seen_at = datetime.datetime.fromisoformat(seen_at)
        target = forecast_hour(seen_at)
        local = pd.DatetimeIndex([target]).tz_convert(LOCAL_TZ)
        return float(self.profile[int(_slot(local)[0])])


def _resolved_frame(refs: typing.List[typing.Any]) -> pd.DataFrame:
    """Every input topic's history as one pandas frame. The driver resolves the
    ObjectRefs; nothing here passes a ref into another task."""
    frames = []
    for ref in refs:
        dataset = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
        frames.append(dataset.to_pandas())
    if not frames:
        return pd.DataFrame(columns=["time", "value"])
    return pd.concat(frames, ignore_index=True)


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit the hourly profile, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        refs = provide_historic_data(TRAINING_WINDOW)
        frame = _resolved_frame(refs)
    if frame.empty or "value" not in frame or "time" not in frame:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("fit"):
        frame = frame[["time", "value"]].dropna()
        times = pd.to_datetime(frame["time"], utc=True)
        series = pd.Series(frame["value"].astype(float).to_numpy(), index=times).sort_index()
        hourly = series.resample("1h").mean().dropna()
        if hourly.empty:
            return None

        slots = _slot(hourly.index.tz_convert(LOCAL_TZ))
        means = hourly.groupby(slots).mean()
        overall = float(hourly.mean())
        profile = [float(means.get(i, overall)) for i in range(48)]

        # In-sample fit of the profile against the hourly means it was built from.
        # A diagnostic, not the evaluation: the evaluation is the replay of the
        # test window, which this fit never sees.
        fitted = pd.Series([profile[int(s)] for s in slots], index=hourly.index)
        train_rmse = float(((hourly - fitted) ** 2).mean() ** 0.5)

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "profile": "weekend flag x local hour (" + LOCAL_TZ + ")",
        "horizon_hours": int(HORIZON.total_seconds() // 3600),
        "hours_fitted": int(len(hourly)),
    })
    logger.log_metrics({"train_rmse_profile": train_rmse})
    return OdeAblationLoadL2HaikuModel(profile=profile)
