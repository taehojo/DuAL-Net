#!/bin/bash

INPUT_BIM="path/to/your/data.bim"
ANNOTATION_CSV="data/snp_annotations.csv"
RAW_DATA="path/to/your/data.raw"
DX_DATA="path/to/your/phenotype.txt"

PREFIX="my_dual_net_run"
RESULT_DIR="results/${PREFIX}"

ANNOTATION_SCRIPT="scripts/annotate_snps.py"
ANALYSIS_SCRIPT="scripts/run_dual_net_analysis.py"

WINDOW_SIZE=100
TABNET_EPOCH=50
TABNET_PATIENCE=10
N_SPLITS=5
ALPHA_TOP_N=100
ALPHA_STEP=0.1
ROC_N_VALUES=(50 100 500 1000 2000)

ANNOTATION_DIR=$(dirname "${ANNOTATION_CSV}")
mkdir -p "${ANNOTATION_DIR}"
mkdir -p "${RESULT_DIR}"

echo "--- Step 1: Running SNP Annotation ---"
echo "Input BIM: ${INPUT_BIM}"
echo "Output CSV: ${ANNOTATION_CSV}"

python "${ANNOTATION_SCRIPT}" "${INPUT_BIM}" "${ANNOTATION_CSV}"

if [ $? -ne 0 ]; then
  echo "Annotation script failed. Exiting pipeline."
  exit 1
fi
echo "Annotation script finished successfully."
echo "----------------------------------------"

echo "--- Step 2: Running DuAL-Net Analysis ---"
echo "Raw data: ${RAW_DATA}"
echo "Phenotype data: ${DX_DATA}"
echo "Annotation data: ${ANNOTATION_CSV}"
echo "Output prefix: ${PREFIX}"
echo "Result directory: ${RESULT_DIR}"
echo "Window size: ${WINDOW_SIZE}"
echo "TabNet Epochs/Patience: ${TABNET_EPOCH}/${TABNET_PATIENCE}"
echo "CV Splits: ${N_SPLITS}"

python "${ANALYSIS_SCRIPT}" \
  "${RAW_DATA}" \
  "${DX_DATA}" \
  "${ANNOTATION_CSV}" \
  --window_size "${WINDOW_SIZE}" \
  --prefix "${PREFIX}" \
  --tabnet_epoch "${TABNET_EPOCH}" \
  --tabnet_patience "${TABNET_PATIENCE}" \
  --n_splits "${N_SPLITS}" \
  --result_dir "${RESULT_DIR}" \
  --roc_n_values ${ROC_N_VALUES[@]} \
  --alpha_top_n "${ALPHA_TOP_N}" \
  --alpha_step "${ALPHA_STEP}"

if [ $? -ne 0 ]; then
  echo "Analysis pipeline failed. Check logs in ${RESULT_DIR}"
  exit 1
fi

echo "Analysis pipeline finished successfully."
echo "Results saved in ${RESULT_DIR}"
echo "----------------------------------------"
echo "Pipeline complete."
