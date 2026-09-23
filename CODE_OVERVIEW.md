# Initial review code (core implementation only)

This package contains the central source files of the progressive physiological guidance method. It is a **source excerpt**, not a runnable training or evaluation release.

- `slowfast/models/contrastive.py` — Video-BYOL student/teacher integration, objective and activation scheduling.
- `slowfast/quasi_periodic/core.py` — temporal adapter, temporal self-similarity, frequency/structural constraints, and background-reference loss.
- `slowfast/quasi_periodic/mask.py` — construction and caching of background masks.
- `slowfast/datasets/` — relevant dataset adapters and background-mask metadata construction.
- `slowfast/config/defaults.py` — configuration schema and default values; **defaults are not the final experiment configuration**.

This subset deliberately omits local/historical YAML configurations, training entrypoints, data preparation, frozen-backbone probe and evaluation scripts, the complete SlowFast dependencies, data, and model weights. It cannot independently reproduce the reported experiments. The core code is based on a SlowFast implementation; upstream attribution and redistribution/license information should be checked before public release.
