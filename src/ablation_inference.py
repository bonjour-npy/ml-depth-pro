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


def run(args):
    """Run Depth Pro on a sample image."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model.
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=(
            torch.half if args.float16 else torch.float
        ),  # 如果参数设置了半精度则使用 torch.half
    )
    model.eval()

    # Export the model to ONNX format with dynamic input size support.
    dummy_input = torch.randn(1, 3, 224, 224, device=get_torch_device()).type(
        torch.half if args.float16 else torch.float
    )
    onnx_output_path = args.output_path / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"}, "output": {0: "batch_size", 2: "height", 3: "width"}},
    )
    print(f"Model has been converted to ONNX and saved at {onnx_output_path}")

    global total_epoch_time
    total_epoch_time = 0.0
    for epoch in range(args.all_epochs):
        # 每个 epoch 重新定义图片迭代器（迭代器会损耗）
        image_paths = [args.image_path]
        if args.image_path.is_dir():
            image_paths = args.image_path.glob("**/*")
            relative_path = args.image_path
        else:
            relative_path = args.image_path.parent

        # 预热 args.warm_up_epochs 个 epoch
        if epoch < args.warm_up_epochs:
            print(f"Epoch {epoch} is warm up epoch, skip inference.")
            for image_path in tqdm(image_paths):
                # Load image and focal length from exif info (if found.).
                try:
                    LOGGER.info(f"Loading image {image_path} ...")
                    image, _, f_px = load_rgb(image_path)
                except Exception as e:
                    LOGGER.error(str(e))
                    continue
                # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
                # otherwise the model estimates `f_px` to compute the depth metricness.
                prediction = model.infer(torch.tensor(transform(image)), f_px=f_px)

        # 正式计算 inference time，执行 args.all_epochs - args.warm_up_epochs 个 epoch
        if epoch >= args.warm_up_epochs:
            print(f"Epoch {epoch} is testing epoch, start inference.")
            # 准备计算 inference time
            torch.cuda.synchronize()  # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程

            # 设置用于测量时间的 cuda Event，这是 PyTorch 官方推荐的接口，理论上应该最靠谱
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )

            per_epoch_time = 0.0  # 每个 epoch 开始时刷新 per_epoch_time
            for image_path in tqdm(image_paths):
                # Load image and focal length from exif info (if found.).
                try:
                    LOGGER.info(f"Loading image {image_path} ...")
                    image, _, f_px = load_rgb(image_path)
                except Exception as e:
                    LOGGER.error(str(e))
                    continue
                # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
                # otherwise the model estimates `f_px` to compute the depth metricness.
                # 开始计时
                starter.record(stream=torch.cuda.current_stream())
                prediction = model.infer(torch.tensor(transform(image)), f_px=f_px)
                ender.record(stream=torch.cuda.current_stream())
                torch.cuda.synchronize()
                # 每个图片的 inference time
                time = starter.elapsed_time(ender)
                per_epoch_time += time

                if epoch == args.all_epochs - 1:
                    print(f"This is the final epoch, save depth map and npz file.")
                    # 保存深度图
                    # Extract the depth and focal length.
                    depth = prediction["depth"].detach().cpu().numpy().squeeze()
                    if f_px is not None:
                        LOGGER.debug(f"Focal length (from exif): {f_px:0.2f}")
                    elif prediction["focallength_px"] is not None:
                        focallength_px = (
                            prediction["focallength_px"].detach().cpu().item()
                        )
                        LOGGER.info(f"Estimated focal length: {focallength_px}")

                    inverse_depth = 1 / depth
                    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
                    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
                    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
                    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                        max_invdepth_vizu - min_invdepth_vizu
                    )

                    # Save Depth as npz file.
                    if args.output_path is not None:
                        output_file = (
                            args.output_path
                            / image_path.relative_to(relative_path).parent
                            / image_path.stem
                        )
                        LOGGER.info(f"Saving depth map to: {str(output_file)}")
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(output_file, depth=depth)

                        # Save as color-mapped "turbo" jpg image.
                        cmap = plt.get_cmap("turbo")
                        color_depth = (
                            cmap(inverse_depth_normalized)[..., :3] * 255
                        ).astype(np.uint8)
                        color_map_output_file = str(output_file) + ".jpg"
                        LOGGER.info(
                            f"Saving color-mapped depth to: : {color_map_output_file}"
                        )
                        PIL.Image.fromarray(color_depth).save(
                            color_map_output_file, format="JPEG", quality=90
                        )

            # 每个 epoch 的 inference time
            print(f"Epoch {epoch} inference time: {per_epoch_time} ms.")
            total_epoch_time += per_epoch_time

    avg_epoch_time = total_epoch_time / (
        float(args.all_epochs) - float(args.warm_up_epochs)
    )
    per_image_time = avg_epoch_time / 525.0
    print(
        f"average inference time: {per_image_time} ms, tested on {args.all_epochs} epochs with {args.warm_up_epochs} warm up epochs."
    )

    LOGGER.info("Done predicting depth!")


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=Path,
        default="./data/example.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default="./output",
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Skip matplotlib display.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output."
    )
    parser.add_argument("-w", "--warm_up_epochs", default=1, help="设置预热的 epoch 数")
    parser.add_argument("-a", "--all_epochs", default=2, help="运行的总 epoch 数")
    parser.add_argument(
        "-f", "--float16", action="store_true", help="使用 float16 半精度进行测试"
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
