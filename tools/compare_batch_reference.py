#!/usr/bin/env python3
"""Compare a threaded --image-dir batch JSON against the 2A/3 reference results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Compare batch JSON with reference results")
    parser.add_argument("--batch-json", required=True, type=Path)
    parser.add_argument("--reference-results", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--conf-tolerance", type=float, default=1e-5)
    parser.add_argument("--bbox-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch = json.loads(args.batch_json.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_results.read_text(encoding="utf-8"))

    ref_thresholds = reference["thresholds"]
    if (
        abs(float(ref_thresholds["conf"]) - args.conf) > 1e-9
        or abs(float(ref_thresholds["iou"]) - args.iou) > 1e-9
    ):
        print("error: CLI thresholds do not match reference thresholds", file=sys.stderr)
        return 2
    if batch["model"]["sha256"] != reference["model"]["sha256"]:
        print("error: batch model SHA256 differs from reference", file=sys.stderr)
        return 2
    if "CPUExecutionProvider" not in batch["model"]["providers"]:
        print("error: batch model did not use CPUExecutionProvider", file=sys.stderr)
        return 2

    ref_by_name = {item["name"]: item for item in reference["images"]}
    expected_names = sorted(ref_by_name)
    frames = batch["frames"]
    if len(frames) != len(expected_names):
        print(
            f"error: batch has {len(frames)} frames, expected {len(expected_names)}",
            file=sys.stderr,
        )
        return 1

    failures = 0
    expected_ok = expected_ng = 0
    max_conf_diff = 0.0
    max_bbox_diff = 0.0
    for index, (frame, name) in enumerate(zip(frames, expected_names)):
        if Path(frame["source"]).name != name:
            print(
                f"frame {index}: reordered: got {Path(frame['source']).name}, "
                f"expected {name}",
                file=sys.stderr,
            )
            failures += 1
            continue
        if int(frame["index"]) != index:
            print(
                f"frame {index}: sequence index is {frame['index']}",
                file=sys.stderr,
            )
            failures += 1
        if frame["error"]:
            print(f"frame {index} ({name}): unexpected ERROR: {frame['error']}", file=sys.stderr)
            failures += 1
            continue

        ref_detections = sorted_detections(ref_by_name[name]["detections"])
        cpp_detections = sorted_detections(frame["detections"])
        expected_status = "NG" if any(
            float(det["confidence"]) > args.conf for det in ref_detections
        ) else "OK"
        if expected_status == "NG":
            expected_ng += 1
        else:
            expected_ok += 1
        if frame["status"] != expected_status:
            print(
                f"frame {index} ({name}): status {frame['status']}, "
                f"expected {expected_status}",
                file=sys.stderr,
            )
            failures += 1
        if not frame.get("quality") or bool(frame["quality"].get("is_ng")) != (
            expected_status == "NG"
        ):
            print(f"frame {index} ({name}): quality field is inconsistent", file=sys.stderr)
            failures += 1

        if len(cpp_detections) != len(ref_detections):
            print(
                f"frame {index} ({name}): detection count {len(cpp_detections)} "
                f"!= {len(ref_detections)}",
                file=sys.stderr,
            )
            failures += 1
            continue
        for cpp, ref in zip(cpp_detections, ref_detections):
            if (
                int(cpp["class_id"]) != int(ref["class_id"])
                or cpp["class_name"] != ref["class_name"]
            ):
                print(f"frame {index} ({name}): class mismatch", file=sys.stderr)
                failures += 1
            conf_diff = abs(float(cpp["confidence"]) - float(ref["confidence"]))
            bbox_diff = max(
                abs(float(a) - float(b))
                for a, b in zip(cpp["bbox_xyxy"], ref["bbox_xyxy"])
            )
            max_conf_diff = max(max_conf_diff, conf_diff)
            max_bbox_diff = max(max_bbox_diff, bbox_diff)
            if conf_diff > args.conf_tolerance or bbox_diff > args.bbox_tolerance:
                print(
                    f"frame {index} ({name}): numeric mismatch "
                    f"conf_diff={conf_diff:.3e} bbox_diff={bbox_diff:.3e}",
                    file=sys.stderr,
                )
                failures += 1

    summary = batch["summary"]
    expected = {
        "total": len(expected_names),
        "ok": expected_ok,
        "ng": expected_ng,
        "errors": 0,
    }
    for key, value in expected.items():
        if int(summary[key]) != value:
            print(
                f"summary {key}={summary[key]}, expected {value}",
                file=sys.stderr,
            )
            failures += 1

    print(
        f"batch comparison: frames={len(frames)} ok={summary['ok']} "
        f"ng={summary['ng']} errors={summary['errors']} "
        f"max_conf_diff={max_conf_diff:.3e} max_bbox_diff={max_bbox_diff:.3e}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
