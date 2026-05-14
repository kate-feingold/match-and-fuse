#!/usr/bin/env bash
# Prepare benchmark datasets for match-and-fuse.
#
# Downloads DreamBooth and/or CustomConcept101, center-crops and resizes images
# so that any pairwise grid has a longest side of 1024 px with both sides
# divisible by 64, then optionally computes RoMa pairwise matches.
#
# Processed images  → data/images/{dataset}/{scene}/
# Pairwise matches  → data/matches/{dataset}/{scene}/
#
# Download and image preparation are skipped if data/images/{dataset}/ already
# has content. Pass --matches to also compute pairwise matches.
#
# Usage:
#   ./scripts/prepare_data.sh --dataset DATASET [OPTIONS]
#
# Options:
#   --dataset  dreambooth | customconcept101 | all  (required)
#   --matches  also compute pairwise matches
#   --device   cuda | cpu                           (default: cuda, for matching)
#   -h, --help
#
# Examples:
#   ./scripts/prepare_data.sh --dataset all
#   ./scripts/prepare_data.sh --dataset dreambooth --matches
#   ./scripts/prepare_data.sh --dataset all --matches --device cpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
RAW_DIR="$DATA_DIR/raw"
IMG_DIR="$DATA_DIR/images"
MATCH_DIR="$DATA_DIR/matches"

DATASET=""
COMPUTE_MATCHES=false
DEVICE="cuda"

usage() {
    sed -n '/^# Usage:/,/^$/p' "$0" | grep -v '^#' | head -n1
    sed -n '/^# Options:/,/^# Examples/p' "$0" | grep '^#' | grep -v 'Examples' | sed 's/^# /  /'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2";        shift 2 ;;
        --matches) COMPUTE_MATCHES=true; shift ;;
        --device)  DEVICE="$2";         shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required (dreambooth | customconcept101 | all)" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$IMG_DIR"

# ---------------------------------------------------------------------------
# Image preparation — skipped if data/images/{dataset}/ is already populated
# ---------------------------------------------------------------------------
if [[ "$DATASET" == "dreambooth" || "$DATASET" == "all" ]]; then
    echo "==> DreamBooth"
    img_dst="$IMG_DIR/dreambooth"
    if [[ -d "$img_dst" && -n "$(ls -A "$img_dst" 2>/dev/null)" ]]; then
        echo "  Already prepared."
    else
        mkdir -p "$RAW_DIR"
        echo "  Cloning..."
        git clone --depth=1 https://github.com/google/dreambooth.git "$RAW_DIR/dreambooth"
        python3 "$SCRIPT_DIR/prepare_images.py" "$RAW_DIR/dreambooth/dataset" "$img_dst"
        rm -rf "$RAW_DIR"
    fi
fi

if [[ "$DATASET" == "customconcept101" || "$DATASET" == "all" ]]; then
    echo "==> CustomConcept101"
    img_dst="$IMG_DIR/customconcept101"
    raw_src="$RAW_DIR/customconcept101"
    if [[ -d "$img_dst" && -n "$(ls -A "$img_dst" 2>/dev/null)" ]]; then
        echo "  Already prepared."
    else
        echo "  Downloading via gdown..."
        mkdir -p "$raw_src"
        cd "$raw_src"
        gdown 1jj8JMtIS5-8vRtNtZ2x8isieWH9yetuK
        unzip -q benchmark_dataset.zip
        rm benchmark_dataset.zip
        cd "$PROJECT_ROOT"
        python3 "$SCRIPT_DIR/prepare_images.py" "$raw_src/benchmark_dataset" "$img_dst"
        rm -rf "$RAW_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Match computation
# ---------------------------------------------------------------------------
if $COMPUTE_MATCHES; then
    mkdir -p "$MATCH_DIR"
    echo "==> Computing pairwise matches"
    if [[ "$DATASET" == "all" ]]; then
        python3 "$SCRIPT_DIR/compute_dataset_matches.py" --img_root "$IMG_DIR" --match_root "$MATCH_DIR" --device "$DEVICE"
    else
        python3 "$SCRIPT_DIR/compute_dataset_matches.py" --img_root "$IMG_DIR/$DATASET" --match_root "$MATCH_DIR/$DATASET" --device "$DEVICE"
    fi
fi

echo ""
echo "Done."
echo "  Processed images : $IMG_DIR/"
if $COMPUTE_MATCHES; then
    echo "  Matches          : $MATCH_DIR/"
fi
