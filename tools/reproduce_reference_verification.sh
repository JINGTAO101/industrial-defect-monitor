#!/usr/bin/env bash
# Reproduce the six-image NEU-DET reference verification from a clean directory.
#
# Downloads (or reuses) the pinned ONNX model, six NEU-DET images and the
# author's src/predict.py at the pinned commit, verifies every SHA256, runs
# tools/neu_det_reference.py for all six images, then compares the results with
# the author's own preprocess_onnx/postprocess_onnx functions.
#
# Prerequisites:
#   - gh, authenticated against github.com (the script uses api.github.com)
#   - curl, sha256sum
#   - a Python environment created from tools/requirements-reference.txt
#
# Usage:
#   VENV="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0/venv" \
#   MODEL_DIR="$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0" \
#   tools/reproduce_reference_verification.sh
#
# The script never writes model weights or test images into the repository.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/edge-ai-models/neu-det-yolov8n-v1.0.0}"
VENV="${VENV:-$MODEL_DIR/venv}"
GH="${GH:-gh}"

MODEL_REPO="winthegame259/steel-surface-defect-detection"
MODEL_TAG="v1.0.0"
MODEL_COMMIT="1226b2d8a241b3ecacb61d9e727051ccb8baa64e"
MODEL_NAME="neu_det_yolov8n_best.onnx"
MODEL_SHA="495ad53dbff99bcbf629ec67f90e1e627800ab8db78177de592648b6e00345b1"

AUTHOR_PREDICT="src/predict.py"
AUTHOR_PREDICT_SHA="84dfc96d0a8bfe9a1b4474b32a2e53c7e94b4ebba2f615269bc5f938b930c130"

IMAGE_REPO="siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection"
IMAGE_COMMIT="351bee21b56bbf97a1e1d18262bceb202eb7ef41"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

verify_sha256() {
    local file="$1" expected="$2"
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        die "SHA256 mismatch for $file: got $actual, expected $expected"
    fi
    echo "  OK  $(basename "$file") $actual"
}

fetch_release_asset() {
    local name="$1" expected="$2"
    local out="$MODEL_DIR/$name"
    if [ -f "$out" ] && [ "$(sha256sum "$out" | awk '{print $1}')" = "$expected" ]; then
        echo "cached $name"
        return
    fi
    rm -f "$out"
    echo "downloading $name via GitHub API"
    local asset_id
    asset_id="$("$GH" api "repos/$MODEL_REPO/releases/tags/$MODEL_TAG" \
        --jq ".assets[] | select(.name==\"$name\") | .id")"
    [ -n "$asset_id" ] || die "release asset not found: $name"
    if ! "$GH" api "repos/$MODEL_REPO/releases/assets/$asset_id" \
        -H "Accept: application/octet-stream" > "$out"; then
        echo "GitHub API asset download failed; falling back to curl"
        curl -fL --retry 2 \
            "https://github.com/$MODEL_REPO/releases/download/$MODEL_TAG/$name" \
            -o "$out"
    fi
    verify_sha256 "$out" "$expected"
}

fetch_image() {
    local name="$1" expected="$2"
    local out="$MODEL_DIR/test-images/$name.jpg"
    if [ -f "$out" ] && [ "$(sha256sum "$out" | awk '{print $1}')" = "$expected" ]; then
        echo "cached $name.jpg"
        return
    fi
    rm -f "$out"
    echo "downloading $name.jpg via GitHub API"
    if ! "$GH" api \
        "repos/$IMAGE_REPO/contents/IMAGES/$name.jpg?ref=$IMAGE_COMMIT" \
        -H "Accept: application/vnd.github.raw" > "$out"; then
        echo "GitHub API image download failed; falling back to raw.githubusercontent.com"
        curl -fL --retry 2 \
            "https://raw.githubusercontent.com/$IMAGE_REPO/$IMAGE_COMMIT/IMAGES/$name.jpg" \
            -o "$out"
    fi
    verify_sha256 "$out" "$expected"
}

fetch_author_source() {
    local out="$MODEL_DIR/reference-src/src_predict.py"
    mkdir -p "$MODEL_DIR/reference-src"
    if [ -f "$out" ] && [ "$(sha256sum "$out" | awk '{print $1}')" = "$AUTHOR_PREDICT_SHA" ]; then
        echo "cached src_predict.py"
        return
    fi
    rm -f "$out"
    echo "downloading author src/predict.py via GitHub API"
    if ! "$GH" api \
        "repos/$MODEL_REPO/contents/$AUTHOR_PREDICT?ref=$MODEL_COMMIT" \
        -H "Accept: application/vnd.github.raw" > "$out"; then
        echo "GitHub API source download failed; falling back to raw.githubusercontent.com"
        curl -fL --retry 2 \
            "https://raw.githubusercontent.com/$MODEL_REPO/$MODEL_COMMIT/$AUTHOR_PREDICT" \
            -o "$out"
    fi
    verify_sha256 "$out" "$AUTHOR_PREDICT_SHA"
}

command -v "$GH" >/dev/null 2>&1 || die "GitHub CLI not found: $GH"
"$GH" auth status >/dev/null 2>&1 || die "GitHub CLI is not authenticated; run: gh auth login"
command -v curl >/dev/null 2>&1 || die "curl not found"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"
[ -x "$VENV/bin/python" ] || die "Python venv not found: $VENV (see tools/requirements-reference.txt)"

mkdir -p \
    "$MODEL_DIR/test-images" \
    "$MODEL_DIR/reference-src" \
    "$MODEL_DIR/verification-results/json" \
    "$MODEL_DIR/verification-results/annotated"

echo "== pinned assets =="
fetch_release_asset "$MODEL_NAME" "$MODEL_SHA"
fetch_author_source

while read -r name expected; do
    [ -n "$name" ] || continue
    fetch_image "$name" "$expected"
done <<'EOF'
crazing_101 0c4ccbfd539aa9d0a151f25b3e798fbad36d2a3532218bcc5c313f06c2c1dda4
scratches_1 500c46e2f5d7065d8c492b0cfb36c3ba77a85a744079bf6453dfc85e02b97a3c
patches_1 0ea4843aa7801ce5a5e672ec8c0266d4910aa7d740955f38d087b38154d0fe67
inclusion_1 2b0a6eff755d5c5867d3e3ca6b26af9548813e50c9b12b0f3381a49560a42634
pitted_surface_1 38a0ece141350f5bed53c99da4d96ce25d42b3ed4248e536a579ab823abfb318
rolled-in_scale_1 e9d0358a2859da86a7a9cc75deb5c235fcb8b14e8773cd92e94593b0289736fd
EOF

echo "== reference inference =="
for name in crazing_101 scratches_1 patches_1 inclusion_1 pitted_surface_1 rolled-in_scale_1; do
    echo "-- $name.jpg"
    "$VENV/bin/python" -I "$REPO_ROOT/tools/neu_det_reference.py" \
        --model "$MODEL_DIR/$MODEL_NAME" \
        --image "$MODEL_DIR/test-images/$name.jpg" \
        --conf 0.25 --iou 0.45 --imgsz 640 \
        --expected-sha256 "$MODEL_SHA" \
        --json-out "$MODEL_DIR/verification-results/json/$name.json" \
        --image-out "$MODEL_DIR/verification-results/annotated/$name.jpg"
done

echo "== author implementation parity =="
"$VENV/bin/python" -I "$REPO_ROOT/tools/compare_author_predict.py" \
    --author-predict "$MODEL_DIR/reference-src/src_predict.py" \
    --model "$MODEL_DIR/$MODEL_NAME" \
    --image-dir "$MODEL_DIR/test-images" \
    --results-dir "$MODEL_DIR/verification-results/json"

echo "REFERENCE_VERIFICATION_OK"
