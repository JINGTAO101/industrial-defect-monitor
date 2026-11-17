#!/usr/bin/env python3
"""Reference CPU inference for the NEU-DET YOLOv8n ONNX model.

The preprocessing and postprocessing here follow the verified reference
implementation in winthegame259/steel-surface-defect-detection at commit
1226b2d8a241b3ecacb61d9e727051ccb8baa64e:
  - src/predict.py :: preprocess_onnx / postprocess_onnx
  - cpp_inference/inference.cpp :: letterbox / infer

Class names are read from the ONNX model metadata; they are never guessed.
Run with `python -I` so the script directory and user site-packages are not
implicitly imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_class_names(session: ort.InferenceSession) -> dict[int, str]:
    """Read the class id -> name mapping from ONNX metadata.

    Raises instead of falling back to a guessed mapping.
    """
    metadata = session.get_modelmeta().custom_metadata_map
    raw = metadata.get("names")
    if not raw:
        raise ValueError("ONNX metadata has no 'names' entry; class mapping cannot be verified")
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"cannot parse ONNX metadata 'names': {raw!r}") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"ONNX metadata 'names' is not a non-empty dict: {parsed!r}")

    names: dict[int, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, int) or not isinstance(value, str):
            raise ValueError(f"invalid class mapping entry: {key!r} -> {value!r}")
        names[key] = value

    expected = list(range(len(names)))
    if sorted(names) != expected:
        raise ValueError(f"class ids are not contiguous from 0: {sorted(names)}")
    return names


def letterbox(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """Resize with unchanged aspect ratio and pad to imgsz x imgsz (gray 114)."""
    orig_h, orig_w = image.shape[:2]
    ratio = imgsz / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = imgsz - new_w
    pad_h = imgsz - new_h
    left = pad_w // 2
    top = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, ratio, left, top


def preprocess(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """BGR -> letterbox -> float32/255 -> NCHW, batch 1."""
    padded, ratio, pad_left, pad_top = letterbox(image, imgsz)
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)  # CHW -> NCHW
    return np.ascontiguousarray(blob), ratio, pad_left, pad_top


def class_aware_nms(
    boxes_xywh: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_thres: float,
    iou_thres: float,
) -> list[int]:
    """Class-aware NMS with OpenCV, matching the reference implementation."""
    keep: list[int] = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes_xywh[cls_mask]
        cls_scores = scores[cls_mask]
        original_indices = np.where(cls_mask)[0]
        nms_indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(), cls_scores.tolist(), conf_thres, iou_thres
        )
        for idx in np.asarray(nms_indices).reshape(-1):
            keep.append(int(original_indices[int(idx)]))
    return keep


def postprocess(
    output: np.ndarray,
    class_names: dict[int, str],
    conf_thres: float,
    iou_thres: float,
    orig_w: int,
    orig_h: int,
    ratio: float,
    pad_left: int,
    pad_top: int,
) -> list[dict]:
    """Decode [1, 4+nc, num_preds] YOLOv8 output into original-image boxes."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"unexpected output shape: {output.shape}")
    num_classes = len(class_names)
    if output.shape[1] != 4 + num_classes:
        raise ValueError(
            f"output channel count {output.shape[1]} != 4 + {num_classes}; "
            "refusing to guess the layout"
        )

    predictions = output[0].T  # [4+nc, num_preds] -> [num_preds, 4+nc]
    boxes = predictions[:, :4]  # cx, cy, w, h in letterboxed input coordinates
    scores = predictions[:, 4:]  # one sigmoid probability per class

    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)
    mask = max_scores > conf_thres
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Undo letterbox padding and scaling.
    x1 = (x1 - pad_left) / ratio
    y1 = (y1 - pad_top) / ratio
    x2 = (x2 - pad_left) / ratio
    y2 = (y2 - pad_top) / ratio

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    keep = class_aware_nms(boxes_xywh, max_scores, class_ids, conf_thres, iou_thres)

    detections: list[dict] = []
    for idx in keep:
        cls_id = int(class_ids[idx])
        if cls_id not in class_names:
            raise ValueError(f"model returned unknown class id {cls_id}")
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "confidence": float(max_scores[idx]),
                "bbox_xyxy": [
                    float(x1[idx]),
                    float(y1[idx]),
                    float(x2[idx]),
                    float(y2[idx]),
                ],
            }
        )
    return detections


def draw(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    colors = [
        (0, 140, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox_xyxy"])
        color = colors[det["class_id"] % len(colors)]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f'{det["class_name"]}: {det["confidence"]:.3f}'
        cv2.putText(
            output,
            label,
            (x1 + 2, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NEU-DET YOLOv8n ONNX CPU reference inference")
    parser.add_argument("--model", required=True, type=Path, help="path to the ONNX model")
    parser.add_argument("--image", required=True, type=Path, help="path to the input image")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="model input size (default 640)")
    parser.add_argument("--expected-sha256", default=None, help="optional expected model SHA256")
    parser.add_argument("--json-out", type=Path, default=None, help="optional JSON result path")
    parser.add_argument("--image-out", type=Path, default=None, help="optional annotated image path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    model_sha = sha256_file(args.model)
    image_sha = sha256_file(args.image)
    if args.expected_sha256 and model_sha.lower() != args.expected_sha256.lower():
        print(
            f"error: model SHA256 mismatch: got {model_sha}, expected {args.expected_sha256}",
            file=sys.stderr,
        )
        return 1

    # CPU only: do not silently fall back to another execution provider.
    if "CPUExecutionProvider" not in ort.get_available_providers():
        print("error: CPUExecutionProvider is unavailable", file=sys.stderr)
        return 1
    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        print(
            f"error: expected 1 input and 1 output, got {len(inputs)} and {len(outputs)}",
            file=sys.stderr,
        )
        return 1
    input_meta = inputs[0]
    output_meta = outputs[0]
    if input_meta.type != "tensor(float)":
        print(f"error: unsupported input type: {input_meta.type}", file=sys.stderr)
        return 1
    if input_meta.shape != [1, 3, args.imgsz, args.imgsz]:
        print(
            f"error: model input {input_meta.name} has shape {input_meta.shape}, "
            f"expected [1, 3, {args.imgsz}, {args.imgsz}]",
            file=sys.stderr,
        )
        return 1

    class_names = read_class_names(session)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: OpenCV failed to read image: {args.image}", file=sys.stderr)
        return 1
    orig_h, orig_w = image.shape[:2]

    blob, ratio, pad_left, pad_top = preprocess(image, args.imgsz)
    raw_output = session.run(None, {input_meta.name: blob})[0]
    detections = postprocess(
        raw_output,
        class_names,
        args.conf,
        args.iou,
        orig_w,
        orig_h,
        ratio,
        pad_left,
        pad_top,
    )

    result = {
        "model": {
            "path": str(args.model),
            "sha256": model_sha,
            "size_bytes": args.model.stat().st_size,
            "metadata": dict(session.get_modelmeta().custom_metadata_map),
        },
        "runtime": {
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "input_contract": {
            "name": input_meta.name,
            "shape": input_meta.shape,
            "type": input_meta.type,
            "layout": "NCHW",
            "color_order": "BGR (cv2.imread)",
            "resize": "letterbox, ratio=imgsz/max(w,h), INTER_LINEAR, pad=114",
            "normalization": "divide by 255 only; no mean/std",
            "batch": 1,
        },
        "output_contract": {
            "name": output_meta.name,
            "shape": output_meta.shape,
            "type": output_meta.type,
            "layout": "channel-first [1, 4+nc, num_preds]",
            "nms_included": False,
        },
        "class_names": {str(k): v for k, v in sorted(class_names.items())},
        "thresholds": {"conf": args.conf, "iou": args.iou},
        "image": {
            "path": str(args.image),
            "sha256": image_sha,
            "width": orig_w,
            "height": orig_h,
            "letterbox": {
                "ratio": ratio,
                "pad_left": pad_left,
                "pad_top": pad_top,
            },
        },
        "detections": detections,
    }

    print(f"ONNX Runtime {ort.__version__} providers={session.get_providers()}")
    print(f"model: {args.model} ({args.model.stat().st_size} bytes)")
    print(f"model sha256: {model_sha}")
    print(f"input:  {input_meta.name} {input_meta.shape} {input_meta.type}")
    print(f"output: {output_meta.name} {output_meta.shape} {output_meta.type}")
    print("classes: " + ", ".join(f"{k}={v}" for k, v in sorted(class_names.items())))
    print(
        f"image: {args.image} {orig_w}x{orig_h} sha256={image_sha} "
        f"letterbox ratio={ratio:.6f} pad=({pad_left},{pad_top})"
    )
    print(f"thresholds: conf>{args.conf} iou={args.iou}")
    print(f"detections: {len(detections)}")
    for det in detections:
        bbox = ", ".join(f"{v:.2f}" for v in det["bbox_xyxy"])
        print(
            f"  class_id={det['class_id']} {det['class_name']} "
            f"conf={det['confidence']:.6f} bbox_xyxy=[{bbox}]"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"json saved: {args.json_out}")
    if args.image_out:
        args.image_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.image_out), draw(image, detections)):
            print(f"error: failed to write annotated image: {args.image_out}", file=sys.stderr)
            return 1
        print(f"annotated image saved: {args.image_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
