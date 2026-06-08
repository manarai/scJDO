#!/usr/bin/env bash
# Re-execute every Figures_notebook under the new kernel-default windowing.
set -u
cd "$(dirname "$0")/.."

# Order: light → heavy
ORDER=(
  "Figures_notebook/Figure2_synthetic_peak_timing_null_analysis.ipynb"
  "Figures_notebook/Figure6_multiome_FA_integration.ipynb"
  "Figures_notebook/Synthetic_benchmark.ipynb"
  "Figures_notebook/08_embedding_benchmark.ipynb"
  "Figures_notebook/Figure5.ipynb"
  "Figures_notebook/09_fa_benchmark.ipynb"
  "Figures_notebook/09_fa_pca_scvi_benchmark.ipynb"
  "Figures_notebook/Figure6_multiome_FA_integration_drift.ipynb"
  "Figures_notebook/Figure3_FA.ipynb"
  "Figures_notebook/Figure4.ipynb"
)

LOG="Figures_notebook/run_all.log"
echo "=== run_all start $(date) ===" > "$LOG"

for nb in "${ORDER[@]}"; do
  name=$(basename "$nb")
  echo "--- $name ---" | tee -a "$LOG"
  t0=$SECONDS
  jupyter nbconvert --to notebook --execute "$nb" \
    --output "$(basename "$nb")" \
    --ExecutePreprocessor.timeout=5400 \
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
