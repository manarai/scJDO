#!/usr/bin/env bash
# Re-execute every Manuscript notebook under the new kernel-default
# windowing. Logs runtime and exit status per notebook.
set -u
cd "$(dirname "$0")/.."

ORDER=(
  "Manuscript/figure_5_updated.ipynb"
  "Manuscript/multiome_autocorrelation_correction.ipynb"
  "Manuscript/figure_2_updated.ipynb"
  "Manuscript/concordance_splicejac_dynamo.ipynb"
  "Manuscript/adaptive_kernel_windowing.ipynb"
  "Manuscript/figure_3_updated.ipynb"
  "Manuscript/figure_4_updated.ipynb"
)

LOG="Manuscript/run_all.log"
echo "=== run_all start $(date) ===" > "$LOG"

for nb in "${ORDER[@]}"; do
  name=$(basename "$nb")
  echo "--- $name ---" | tee -a "$LOG"
  t0=$SECONDS
  jupyter nbconvert --to notebook --execute "$nb" \
    --output "$(basename "$nb")" \
    --ExecutePreprocessor.timeout=3600 \
    >> "$LOG" 2>&1
  rc=$?
  dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "OK   $name  ${dt}s" | tee -a "$LOG"
  else
    echo "FAIL $name  ${dt}s  rc=$rc" | tee -a "$LOG"
  fi
done

echo "=== run_all done $(date) ===" >> "$LOG"
