# MachinaSense Studio — responsive local edition

A responsive frontend and FastAPI backend inspired by [SubhasisRoy578/MachinaSense](https://github.com/SubhasisRoy578/MachinaSense). It includes a dashboard, CSV dataset explorer, actual scikit-learn condition-classifier training, local model persistence, prediction UI, a simple temperature-threshold screen, and API health/model endpoints.

## Important model distinction

The upstream repository's documented bundled benchmark models are trained for **NASA C-MAPSS turbofan remaining-useful-life (RUL)** prediction. They are not a bearing healthy-vs-faulty classifier, and their output/feature schema is not interchangeable with bearing vibration features. This edition does not pretend otherwise. It trains a separate classifier from a labelled CSV you provide. To run the upstream C-MAPSS benchmark, use the original repository and its complete `backend/artifacts` and backend source; its binary model artifacts are not included in this package.

This is a useful local starting point, not a byte-for-byte copy of every upstream service. The original project includes Clerk authentication, per-user tenancy, PostgreSQL persistence, model versioning/activation, training-job history, RAG, copilot integrations, and additional diagnostics. This local edition intentionally implements a smaller set of core ML workflows. See the upstream repo for its complete features and license.

## Run on Windows

1. Extract this ZIP to a folder.
2. Install Python 3.10+ if needed.
3. Double-click `run_windows.bat`.
4. Wait for package installation. Open **http://127.0.0.1:8000** in Chrome.
5. Keep the terminal window open while using the site.

The first launch requires internet to install Python packages. If Windows blocks the script, open PowerShell in the extracted folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

If PowerShell refuses activation, use `run_windows.bat` or run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload`.

## CSV format for training

Use one row per observation/window and a categorical target column, for example:

```csv
rms,kurtosis,peak_frequency,rpm,load,condition
0.21,3.1,120,1797,0,healthy
0.25,3.4,122,1797,0,healthy
0.86,7.8,145,1797,0,faulty
0.92,8.2,149,1797,0,faulty
```

These values are illustrative only, not real bearing measurements. Choose feature columns that match the data used for the task, and choose `condition` as the target. For serious evaluation, keep windows from the same original recording/machine in the same split; the current simple random split is not group-aware and may overstate performance if windows from one recording are mixed across train/test.

## Temperature

The temperature page is an explicit-threshold screen only. Set thresholds based on manufacturer guidance and validated operating limits. It is not calibrated to a bearing, and temperature is not included in the trained classifier unless you include it as a feature in properly paired, labelled training data.

## Security

The server binds to `127.0.0.1` by default. Do not expose it publicly as-is. Before deploying, add authentication/authorization, CSRF protections where relevant, rate limits, upload validation, production CORS settings, persistent database storage, and a production server configuration.
