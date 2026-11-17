#!/usr/bin/env python3
"""Compare the C++ inference output with the 2A Python reference results.

For every JPG in --image-dir:
  1. run edge_ai with --json-out and --dump-input;
  2. compare detections (count, class, confidence, bbox) with the matching
     entry in the committed docs/reference-inference-results.json;
  3. recompute the Python preprocessing blob and compare it with the C++
     raw NCHW float32 dump.

Exit code 0 only when all images pass all tolerances.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def load_reference_preprocess():
    """Load the exact 2A reference implementation from tools/neu_det_reference.py."""
    path = Path(__file__).with_name("neu_det_reference.py")
    spec = importlib.util.spec_from_file_location("neu_det_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference preprocessing script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess


def sorted_detections(detections: list[dict]) -> list[dict]:
    return sorted(
        detections,
        key=lambda d: (
            int(d["class_id"]),
            -float(d["confidence"]),
            float(d["bbox_xyxy"][0]),
            float(d["bbox_xyxy"][1]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare C++ and 2A reference inference")
    parser.add_argument("--edge-ai", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--reference-results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--conf-tolerance", type=float, default=1e-5)
    parser.add_argument("--bbox-tolerance", type=float, default=1e-4)
    parser.add_argument("--blob-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.edge_ai.is_file():
        print(f"error: edge_ai not found: {args.edge_ai}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2
    reference_data = json.loads(args.reference_results.read_text(encoding="utf-8"))
    reference_by_name = {item["name"]: item for item in reference_data["images"]}
    reference_sha = reference_data["model"]["sha256"]
    print(f"reference model sha256: {reference_sha}")
    reference_preprocess = load_reference_preprocess()

    images = sorted(args.image_dir.glob("*.jpg"))
    if not images:
        print(f"error: no JPG images in {args.image_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    max_conf_diff = 0.0
    max_bbox_diff = 0.0
    max_blob_diff = 0.0

    for image_path in images:
        reference = reference_by_name.get(image_path.name)
        if reference is None:
            print(f"{image_path.name}: FAIL: no reference entry", file=sys.stderr)
            failures += 1
            continue

        stem = image_path.stem
        cpp_json = args.out_dir / f"{stem}.cpp.json"
        cpp_dump = args.out_dir / f"{stem}.cpp.bin"
        command = [
            str(args.edge_ai),
            "--model",
            str(args.model),
            "--image",
            str(image_path),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--json-out",
            str(cpp_json),
            "--dump-input",
            str(cpp_dump),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print(
                f"{image_path.name}: FAIL: edge_ai rc={completed.returncode}: "
                f"{completed.stderr.strip()}",
                file=sys.stderr,
            )
            failures += 1
            continue

        cpp_data = json.loads(cpp_json.read_text(encoding="utf-8"))
        if cpp_data["model"]["sha256"] != reference_sha:
            print(
                f"{image_path.name}: FAIL: model SHA256 differs from the 2A reference",
                file=sys.stderr,
            )
            failures += 1
            continue
        if "CPUExecutionProvider" not in cpp_data["model"]["providers"]:
            print(
                f"{image_path.name}: FAIL: CPUExecutionProvider is not reported",
                file=sys.stderr,
            )
            failures += 1
            continue
        if (
            abs(float(cpp_data["thresholds"]["conf"]) - args.conf) > 1e-6
            or abs(float(cpp_data["thresholds"]["iou"]) - args.iou) > 1e-6
        ):
            print(
                f"{image_path.name}: FAIL: thresholds differ from requested values",
                file=sys.stderr,
            )
            failures += 1
            continue
        cpp_detections = sorted_detections(cpp_data["detections"])
        ref_detections = sorted_detections(reference["detections"])

        detections_ok = len(cpp_detections) == len(ref_detections)
        class_ok = detections_ok
        conf_diff = 0.0
        bbox_diff = 0.0
        if detections_ok:
            for cpp, ref in zip(cpp_detections, ref_detections):
                class_ok &= (
                    int(cpp["class_id"]) == int(ref["class_id"])
                    and cpp["class_name"] == ref["class_name"]
                )
                conf_diff = max(
                    conf_diff,
                    abs(float(cpp["confidence"]) - float(ref["confidence"])),
                )
                bbox_diff = max(
                    bbox_diff,
                    max(
                        abs(float(a) - float(b))
                        for a, b in zip(cpp["bbox_xyxy"], ref["bbox_xyxy"])
                    ),
                )
        max_conf_diff = max(max_conf_diff, conf_diff)
        max_bbox_diff = max(max_bbox_diff, bbox_diff)

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{image_path.name}: FAIL: cannot read image", file=sys.stderr)
            failures += 1
            continue
        reference_blob, _, _, _ = reference_preprocess(image, 640)
        cpp_blob = np.fromfile(cpp_dump, dtype=np.float32)
        expected_elements = reference_blob.size
        if cpp_blob.size != expected_elements:
            print(
                f"{image_path.name}: FAIL: C++ blob has {cpp_blob.size} floats, "
                f"expected {expected_elements}",
                file=sys.stderr,
            )
            failures += 1
            continue
        cpp_blob = cpp_blob.reshape(reference_blob.shape)
        blob_diff = float(np.max(np.abs(reference_blob - cpp_blob)))
        max_blob_diff = max(max_blob_diff, blob_diff)

        ok = (
            detections_ok
            and class_ok
            and conf_diff <= args.conf_tolerance
            and bbox_diff <= args.bbox_tolerance
            and blob_diff <= args.blob_tolerance
        )
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"{image_path.name}: {status}: "
            f"detections cpp/ref={len(cpp_detections)}/{len(ref_detections)} "
            f"class_ok={class_ok} conf_max_diff={conf_diff:.3e} "
            f"bbox_max_diff={bbox_diff:.3e} blob_max_diff={blob_diff:.3e} "
            f"(tolerances conf={args.conf_tolerance:g} "
            f"bbox={args.bbox_tolerance:g} blob={args.blob_tolerance:g})"
        )

    print(
        f"SUMMARY: images={len(images)} failures={failures} "
        f"max_conf_diff={max_conf_diff:.3e} max_bbox_diff={max_bbox_diff:.3e} "
        f"max_blob_diff={max_blob_diff:.3e}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
