# type: ignore

import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm  # type: ignore
import onnxruntime as ort

from depth_pro import load_rgb

from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)


LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"\tUsing device: {device}")
    return device


def run(args):
    """Run Depth Pro on a sample image."""

    providers = (
        ["CUDAExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load ONNX model.
    ort_session = ort.InferenceSession(str(args.onnx_model_path), providers=providers)

    # 定义 transform
    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(get_torch_device())),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            (
                ConvertImageDtype(torch.float32)
                # if not args.half_float
                # else ConvertImageDtype(torch.float16)
            ),
        ]
    )

    global total_epoch_time
    total_epoch_time = 0.0
    for epoch in range(args.all_epochs):
        # 每个 epoch 重新定义图片迭代器（迭代器会损耗）
        image_paths = [args.image_path]
        if args.image_path.is_dir():
            # 只对测试文件中文件名中包含 '09_28' 的 20 张图片进行测试
            image_paths = [p for p in args.image_path.glob("**/*") if "09_28" in p.name]
            image_counter = len(image_paths)
            # 将 image_paths 转为迭代器
            image_paths = iter(image_paths)
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
                    image, _, f_px = load_rgb(image_path)  # image: [H, W, C]
                except Exception as e:
                    LOGGER.error(str(e))
                    continue
                # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
                # otherwise the model estimates `f_px` to compute the depth metricness.
                input_tensor = (
                    transform(image)
                    .unsqueeze(0)
                    .cpu()
                    .numpy()
                    # .astype(np.float32 if not args.half_float else np.float16)
                )  # Add batch dimension
                ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
                ort_outs = ort_session.run(None, ort_inputs)
                prediction = {
                    "depth": torch.tensor(ort_outs[0]),
                    "focallength_px": f_px,
                }

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
                input_tensor = (
                    transform(image)
                    .unsqueeze(0)
                    .cpu()
                    .numpy()
                    # .astype(np.float32 if not args.half_float else np.float16)
                )  # Add batch dimension
                ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
                # 开始计时
                starter.record(stream=torch.cuda.current_stream())
                ort_outs = ort_session.run(None, ort_inputs)
                # 结束计时
                ender.record(stream=torch.cuda.current_stream())
                prediction = {
                    "depth": torch.tensor(ort_outs[0]),
                    "focallength_px": f_px,
                }
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

                    # 取消深度倒数的操作，直接保存深度图

                    # inverse_depth = 1 / depth
                    # # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
                    # max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
                    # min_invdepth_vizu = max(1 / 250, inverse_depth.min())
                    # inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                    #     max_invdepth_vizu - min_invdepth_vizu
                    # )

                    # Save Depth as npz file.
                    if args.output_path is not None:
                        if args.half_float:
                            output_file = (
                                args.output_path
                                / "fp16"
                                / image_path.relative_to(relative_path).parent
                                / image_path.stem
                            )
                        else:
                            output_file = (
                                args.output_path
                                / "fp32"
                                / image_path.relative_to(relative_path).parent
                                / image_path.stem
                            )
                        LOGGER.info(f"Saving depth map to: {str(output_file)}")
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(output_file, depth=depth)

                        # 取消深度倒数的操作，直接保存深度图

                        # # Save as color-mapped "turbo" jpg image.
                        # cmap = plt.get_cmap("turbo")
                        # color_depth = (
                        #     cmap(inverse_depth_normalized)[..., :3] * 255
                        # ).astype(np.uint8)
                        # color_map_output_file = str(output_file) + ".png"
                        # LOGGER.info(
                        #     f"Saving color-mapped depth to: : {color_map_output_file}"
                        # )
                        # PIL.Image.fromarray(color_depth).save(
                        #     color_map_output_file, format="PNG"
                        # )

                        # 保存深度的灰度图
                        depth_output_file = str(output_file) + ".png"  # 保存深度图
                        LOGGER.info(f"Saving depth map to: {depth_output_file}")
                        PIL.Image.fromarray((depth * 256).astype(np.uint16)).save(
                            depth_output_file, format="PNG"
                        )

                    print(f"Depth map and npz file saved.")

            # 每个 epoch 的 inference time
            print(f"Epoch {epoch} inference time: {per_epoch_time} ms.")
            total_epoch_time += per_epoch_time

    avg_epoch_time = total_epoch_time / (
        float(args.all_epochs) - float(args.warm_up_epochs)
    )
    per_image_time = avg_epoch_time / float(image_counter)
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
        default="./data/depth/depth_selection/val_selection_cropped/image",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default="./outputs/val_selection_cropped/onnx",
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
    parser.add_argument("-a", "--all_epochs", default=3, help="运行的总 epoch 数")
    parser.add_argument(
        "--half-float", action="store_true", help="使用 float16 半精度进行测试"
    )
    parser.add_argument(
        "--onnx-model-path",
        type=Path,
        default="./checkpoints/onnx/model.onnx",
        help="Path to the ONNX model file.",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
