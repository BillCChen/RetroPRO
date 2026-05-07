#!/usr/bin/env bash
set -euo pipefail

APP_URL="${APP_URL:-http://127.0.0.1:18100}"
EDGE_URL="${EDGE_URL:-http://127.0.0.1}"
RUN_PREDICT="false"

if [[ "${1:-}" == "--run-predict" ]]; then
  RUN_PREDICT="true"
fi

echo "[check] app root: ${APP_URL}/"
curl -fsS "${APP_URL}/" >/dev/null

echo "[check] preview-smiles"
preview_resp="$(curl -fsS -X POST "${APP_URL}/api/preview-smiles" \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCOCC"}')"

python - <<'PY' "${preview_resp}"
import json, sys
obj = json.loads(sys.argv[1])
if not obj.get("valid"):
    raise SystemExit("preview-smiles returned invalid result")
print("[ok] preview-smiles valid")
PY

echo "[check] edge root: ${EDGE_URL}/"
curl -fsS "${EDGE_URL}/" >/dev/null

if [[ "${RUN_PREDICT}" == "true" ]]; then
  echo "[check] predict flow (this may take time)"
  predict_resp="$(curl -fsS -X POST "${APP_URL}/api/predict" \
    -H 'Content-Type: application/json' \
    -d '{"smiles":"CCOCC","iterations":30,"expansion_topk":8,"use_value_fn":true,"one_step_type":"mlp","CCS":true,"radius":9}')"

  task_id="$(python - <<'PY' "${predict_resp}"
import json, sys
obj = json.loads(sys.argv[1])
task_id = obj.get("task_id")
if not task_id:
    raise SystemExit("predict did not return task_id")
print(task_id)
PY
  )"

  echo "[check] result: ${task_id}"
  curl -fsS "${APP_URL}/api/result/${task_id}" >/dev/null

  echo "[check] html download: ${task_id}"
  curl -fsS "${APP_URL}/api/download_html/${task_id}" >/dev/null
fi

echo "[ok] Plan A smoke test passed"
