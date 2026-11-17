#!/usr/bin/env python3
"""Generate deterministic, synthetic test fixtures for task 2B.

These files contain no third-party images or trained weights:
  - non_square_320x200.png: synthetic RGB gradient with two shapes
  - wrong_input.onnx: valid ONNX, but input shape is not [1,3,640,640]
  - wrong_output.onnx: valid ONNX, but output shape is not [1,10,8400]
  - corrupt_model.bin: deliberately invalid ONNX bytes

Run: python -I tools/generate_test_fixtures.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image, ImageDraw

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def make_image() -> None:
    width, height = 320, 200
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 255) // (width - 1),
                (y * 255) // (height - 1),
                128,
            )
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, width - 10, height - 10), fill=(255, 255, 255), width=2)
    draw.ellipse(
        (width // 2 - 20, height // 2 - 20, width // 2 + 20, height // 2 + 20),
        outline=(0, 0, 0),
        width=2,
    )
    image.save(FIXTURES / "non_square_320x200.png")


def make_wrong_contract_model(path: Path, input_shape: list[int], output_shape: list[int]) -> None:
    """Create a loadable ONNX model whose declared contract is deliberately wrong."""
    graph_input = helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)
    graph_output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)
    values = np.zeros(int(np.prod(output_shape)), dtype=np.float32).tobytes()
    constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=helper.make_tensor(
            "output0_value", TensorProto.FLOAT, output_shape, values, raw=True
        ),
    )
    graph = helper.make_graph([constant], "wrong_contract", [graph_input], [graph_output])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 12)],
        producer_name="edge-ai-test-fixture",
    )
    model.ir_version = 7
    names_entry = model.metadata_props.add()
    names_entry.key = "names"
    names_entry.value = (
        "{0: 'crazing', 1: 'inclusion', 2: 'patches', "
        "3: 'pitted_surface', 4: 'rolled-in_scale', 5: 'scratches'}"
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_zero_output_model(path: Path) -> None:
    """Create a valid-contract model whose output is all zeros (empty detections)."""
    graph_input = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    graph_output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 10, 8400])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    target_shape = helper.make_tensor(
        "target_shape", TensorProto.INT64, [3], [1, 10, 8400]
    )
    expand = helper.make_node("Expand", ["zero", "target_shape"], ["output0"])
    graph = helper.make_graph(
        [expand],
        "zero_output",
        [graph_input],
        [graph_output],
        initializer=[zero, target_shape],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 12)],
        producer_name="edge-ai-test-fixture",
    )
    model.ir_version = 7
    names_entry = model.metadata_props.add()
    names_entry.key = "names"
    names_entry.value = (
        "{0: 'crazing', 1: 'inclusion', 2: 'patches', "
        "3: 'pitted_surface', 4: 'rolled-in_scale', 5: 'scratches'}"
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_image()
    make_wrong_contract_model(
        FIXTURES / "wrong_input.onnx", [1, 3, 320, 320], [1, 10, 1]
    )
    make_wrong_contract_model(
        FIXTURES / "wrong_output.onnx", [1, 3, 640, 640], [1, 10, 100]
    )
    make_zero_output_model(FIXTURES / "zero_output.onnx")
    (FIXTURES / "corrupt_model.bin").write_bytes(b"NOT_AN_ONNX_MODEL\n")
    for path in sorted(FIXTURES.iterdir()):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
