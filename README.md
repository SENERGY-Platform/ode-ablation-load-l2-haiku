# ode-ablation-load-l2-haiku

## Summary: net grid consumption, 24 hours ahead

**Problem.** Forecast net electrical consumption from the grid, 24 hours ahead, as the hourly mean power in watts. The evaluation (`evaluation.yaml`) scores the RMSE of the hourly predictions against the hourly mean of the target over a held-out test window, with a threshold of 6.1 W, lower is better.

**Outcome.** The criterion is not met. The best evaluated result is a test RMSE of 219.5 W, about 36 times the threshold. The operator in this repository is a baseline. It is not a finished forecaster.

### Data

- The ontology search for grid power returned no devices. The device list did. The series used is `sensor.MT175.P` of the Iskra MT 175 meter (`Stromzähler`), device `urn:infai:ses:device:8ae74f8f-07c9-42c2-ad1c-701b503adac1`, service `urn:infai:ses:service:61447a7c-5e75-48dd-b2db-bcfcd23fa33d`. It is signed active power in W, sampled about every 10 seconds, with history from 2024-11-28. Negative values are feed-in, so it is the net grid series.
- The developer confirmed this series as the data selection. The evaluation's target is withheld from the operator, but the metric resolved against this single input mapping, which indicates the target is this series.
- Other candidates were not used. The Qubino three-phase meter has a long history and is irregularly sampled, and its role on the site is unconfirmed. The APSystems inverter was offline. The weather imports (Open-Meteo history, which is actuals rather than forecasts, and the yr.no 48-hour forecast) were not used, because the split hides test-period weather values from the session, so a weather-based forecast could not be evaluated honestly.
- The energy counters on the meter (`E_in`, `E_out`) disagree with the power channel and show frozen runs, so only the power channel is used.

### Approach

1. **Training.** `training.py` fits a profile of net grid power: the mean hourly power for each combination of weekend flag and local hour of day (Europe/Berlin), over the 90 days before the split. Local time is an assumption, based on the German device names and the weather location, and has not been confirmed by the site owner.
2. **Forecast.** `op.py` emits, for each message at time t, the profile value for the hour that starts 24 hours after t, timestamped with that hour. The evaluation buckets predictions by that timestamp, so each hour gets an average over the messages that forecast it.
3. **Scaffold fix.** The scaffold's training passed Ray object references into a remote task without resolving them, so training would have failed. `training.py` now resolves them in the driver.

### Results

Evaluation over 2026-09-01 to 2026-10-01 (648 scored hours), from the MLflow runs of this repository:

| Launch | Change | Test RMSE (W) |
|---|---|---|
| 1 | 90-day profile, no offset (commit `c26f46b`) | 219.5 |
| 2 | 28-day profile (`52fc630`) | 229.5 |
| 3 | 90-day profile with per-hour training diagnostics (`cf0e80c`) | 219.5 |
| 4 | Plus offset from all 24 previous hours (`873c879`) | 229.4 |
| 5 | Plus offset from the same hour on the previous 7 days (`043f2df`) | 229.8 |

Launch 3 reproduces launch 1 exactly. The committed `op.py` at `a90056b` restores the launch 3 behaviour; confirm that with `git diff cf0e80c a90056b -- op.py`.

The in-sample profile error by local hour shows where the baseline fails. It is 33 to 59 W from midnight to 05:00, 117 to 206 W from 07:00 to 09:00, 250 to 283 W from 10:00 to 15:00, and 325 to 404 W from 16:00 to 19:00. The profile does not represent the day's variation, and the daytime and evening hours carry most of the error.

### Why the criterion is not met

- **The profile has no input for within-day variation.** Weather, PV output and recent level are all absent. The two level corrections tried both raised the error by about 10 W, so the level residual on this series does not carry into the hours being forecast.
- **The threshold may be unreachable.** The grid meter's hourly means vary by hundreds of watts, and no 24-hour-ahead forecast from this platform's data is likely to get near 6 W. The unit and horizon of the threshold should be confirmed before further work. The criterion is the developer's to set, and it is not changed here.

### Caveats

- **The September window has been used for selection.** Five launches were compared on the same test window, and the later variants were chosen partly by their test scores. The 219.5 W figure is therefore somewhat optimistic, and a fresh window is needed to judge generalisation.
- **Profile statistics are partial.** The profiler's distribution figures covered only the last 36 hours of its window, because of a raw-row cap, so they are not a description of the three months.
- **The operator reads the training window only through the platform.** No value after 2026-09-01 is visible to the session, so none of the test-window figures here were checked against the raw data by hand.

### What would be needed to go further

- A forecast of weather or PV output for the forecast hours, from a historical forecast source. This platform does not provide one for the test period under the current split.
- A decision on the threshold, and on whether the test window should be replaced.
- A recent-level input that is validated on training data only, so that the test window is not used to choose it.

---

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.2". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-load-l2-haiku". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-load-l2-haiku:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.2", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
