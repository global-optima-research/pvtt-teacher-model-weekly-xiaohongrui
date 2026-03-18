#!/bin/bash
# ===========================================================================
# PVTT Evaluation: Compute all metrics for FFGo video generation results
#
# Computes FiVE-Bench, Edit Success, and VBench metrics for generated videos.
#
# Usage:
#   bash scripts/run_pvtt_evaluate.sh
#
#   # Custom generated directory:
#   GENERATED_DIR=experiments/results/ffgo_original/pvtt/20260317_185832 \
#       bash scripts/run_pvtt_evaluate.sh
#
#   # Evaluate specific tasks:
#   TASK_IDS="0016-bracelet1_to_bracelet3,0021-earring1_to_earring3" \
#       bash scripts/run_pvtt_evaluate.sh
#
#   # Change skip frames:
#   SKIP_FRAMES=0 bash scripts/run_pvtt_evaluate.sh
# ===========================================================================

set -e

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- Path Configuration ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="$PROJECT_DIR/pvtt_evaluation/metrics/evaluate_pvtt.py"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/samples/pvtt_evaluation_datasets}"

# --- Find the latest generated results directory if not specified ---
if [ -z "$GENERATED_DIR" ]; then
    RESULTS_BASE="$PROJECT_DIR/experiments/results/ffgo_original/pvtt"
    if [ -d "$RESULTS_BASE" ]; then
        GENERATED_DIR=$(ls -dt "$RESULTS_BASE"/*/ 2>/dev/null | head -1)
        GENERATED_DIR="${GENERATED_DIR%/}"  # Remove trailing slash
    fi
fi

if [ -z "$GENERATED_DIR" ] || [ ! -d "$GENERATED_DIR" ]; then
    echo "ERROR: Generated results directory not found."
    echo "  Set GENERATED_DIR or ensure experiments/results/ffgo_original/pvtt/ exists."
    exit 1
fi

# --- Evaluation Parameters ---
SKIP_FRAMES="${SKIP_FRAMES:-4}"
EVAL_H="${EVAL_H:-480}"
EVAL_W="${EVAL_W:-832}"
VIDEO_FILENAME="${VIDEO_FILENAME:-ffgo_original.mp4}"
REF_FRAME_FILENAME="${REF_FRAME_FILENAME:-ffgo_original_ref_frame.jpg}"

# --- Output ---
OUTPUT_CSV="${OUTPUT_CSV:-$GENERATED_DIR/evaluation_results.csv}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-$GENERATED_DIR/evaluation_summary.json}"

# --- Optional: task IDs filter ---
TASK_IDS_FLAG=""
if [ -n "$TASK_IDS" ]; then
    TASK_IDS_FLAG="--task_ids $TASK_IDS"
fi

# --- Optional: verbose ---
VERBOSE_FLAG=""
if [ "${VERBOSE:-0}" = "1" ]; then
    VERBOSE_FLAG="--verbose"
fi

# --- Optional: custom JSON path ---
JSON_FLAG=""
if [ -n "$JSON_PATH" ]; then
    JSON_FLAG="--json_path $JSON_PATH"
fi

# ===========================================================================
echo "============================================================"
echo "PVTT Evaluation"
echo "============================================================"
echo "Generated dir:   $GENERATED_DIR"
echo "Dataset root:    $DATASET_ROOT"
echo "Skip frames:     $SKIP_FRAMES"
echo "Eval resolution: ${EVAL_W}x${EVAL_H}"
echo "Video filename:  $VIDEO_FILENAME"
echo "Output CSV:      $OUTPUT_CSV"
echo "Output summary:  $OUTPUT_SUMMARY"
echo "============================================================"

python3 "$EVAL_SCRIPT" \
    --generated_dir "$GENERATED_DIR" \
    --dataset_root "$DATASET_ROOT" \
    --output_csv "$OUTPUT_CSV" \
    --output_summary "$OUTPUT_SUMMARY" \
    --video_filename "$VIDEO_FILENAME" \
    --ref_frame_filename "$REF_FRAME_FILENAME" \
    --skip_frames "$SKIP_FRAMES" \
    --eval_h "$EVAL_H" \
    --eval_w "$EVAL_W" \
    $JSON_FLAG \
    $TASK_IDS_FLAG \
    $VERBOSE_FLAG

EXIT_CODE=$?
echo ""
echo "Evaluation finished, exit code: $EXIT_CODE"
exit $EXIT_CODE
