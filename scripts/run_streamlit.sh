#!/usr/bin/env bash
set -euo pipefail
export DEFECTVISION_DEPLOY_ROOT="${DEFECTVISION_DEPLOY_ROOT:-$PWD/deploy}"
python -m streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
