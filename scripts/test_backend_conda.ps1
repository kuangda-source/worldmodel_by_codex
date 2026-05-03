$ErrorActionPreference = "Stop"

conda run -n orwm311 python -m pytest backend/tests
