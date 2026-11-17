#!/usr/bin/env python3
"""Compare tools/neu_det_reference.py results with the model author's predict.py.

This loads the author's `src/predict.py` (downloaded separately, AGPL-3.0) and
calls its `preprocess_onnx` / `postprocess_onnx` functions directly. The
`ultralytics` import at the top of that file is stubbed because only the ONNX
path is exercised.

Usage:
  python -I tools/compare_author_predict.py \
    --author-predict /path/to/src/predict.py \
    --model /path/to/neu_det_yolov8n_best.onnx \
    --image-dir /path/to/test-images \
    --results-dir /path/to/verification-results/json

Exit code 0 means all detections are bit-identical.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_author_module(path: Path):
    stub = types.ModuleType("ultralytics")
    stub.YOLO = object
    sys.modules["ultralytics"] = stub

    spec = importlib.util.spec_from_file_location("author_predict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load author module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare our reference inference with the author's predict.py")
    parser.add_argument("--author-predict", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    for path in (args.author_predict, args.model, args.image_dir, args.results_dir):
        if not path.exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2

    author = load_author_module(args.author_predict)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    images = sorted(args.image_dir.glob("*.jpg"))
    if not images:
        print(f"error: no .jpg files in {args.image_dir}", file=sys.stderr)
        return 2

    all_ok = True
    for image_path in images:
        result_path = args.results_dir / f"{image_path.stem}.json"
        if not result_path.is_file():
            print(f"error: missing result JSON: {result_path}", file=sys.stderr)
            return 2
        ours = json.loads(result_path.read_text(encoding="utf-8"))["detections"]

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"error: cannot read image: {image_path}", file=sys.stderr)
            return 2
        orig_h, orig_w = image.shape[:2]
        blob, ratio, pad = author.preprocess_onnx(image, args.imgsz)
        raw = session.run(None, {input_name: blob})[0]
        reference = author.postprocess_onnx(
            raw, args.conf, args.iou, orig_w, orig_h, ratio, pad
        )

        same_count = len(reference) == len(ours)
        max_diff = 0.0
        same_class_ids = True
        if same_count:
            for ref, own in zip(reference, ours):
                same_class_ids &= ref["class_id"] == own["class_id"]
                max_diff = max(max_diff, abs(ref["confidence"] - own["confidence"]))
                max_diff = max(
                    max_diff,
                    max(abs(a - b) for a, b in zip(ref["bbox"], own["bbox_xyxy"])),
                )
        ok = same_count and same_class_ids and max_diff == 0.0
        all_ok &= ok
        print(
            f"{image_path.name}: author={len(reference)} ours={len(ours)} "
            f"class_ids_match={same_class_ids} max_abs_diff={max_diff:.3e} "
            f"-> {'IDENTICAL' if ok else 'DIFF'}"
        )

    print("PARITY:", "IDENTICAL" if all_ok else "MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
