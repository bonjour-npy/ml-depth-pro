# type: ignore

import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm  # type: ignore

from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")
    return device


def export_onnx(args):
    with torch.no_grad():
        # Load model.
        model, transform = create_model_and_transforms(
            device=get_torch_device(),
            precision=(
                torch.half if args.half_float else torch.float
            ),  # 如果参数设置了半精度则使用 torch.half
        )
        model.eval()

        if args.half_float:
            model.half()

        # -------------------------------------------------------------------------- #
        # Export the model to ONNX format with dynamic input size support.
        dummy_input = torch.randn(1, 3, 1536, 1536, device=get_torch_device()).type(
            torch.half if args.half_float else torch.float
        )
        if args.half_float:
            dummy_input = dummy_input.half()
        if args.half_float:
            onnx_output_path = args.output_path / "model_fp16.onnx"
            onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            onnx_output_path = args.output_path / "model.onnx"
            onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_output_path = str(onnx_output_path)  # Ensure the path is a string
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            # large_model=True,  # Ensure large_model is set to True
        )
        print(f"Model has been converted to ONNX and saved at {onnx_output_path}")
        # -------------------------------------------------------------------------- #


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default="./onnx_weights",
        help="Path to store output files.",
    )
    parser.add_argument(
        "--half-float", action="store_true", help="使用 float16 半精度导出"
    )

    export_onnx(parser.parse_args())


if __name__ == "__main__":
    main()
