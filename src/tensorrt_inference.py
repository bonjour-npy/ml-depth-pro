# type: ignore

import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm  # type: ignore
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

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
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"\tUsing device: {device}")
    return device


def load_engine(engine_file_path):
    """Load a TensorRT engine from file."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    """Allocate host and device buffers for TensorRT engine inference."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    """Perform inference with TensorRT."""
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return [out['host'] for out in outputs]


def run(args):
    """Run Depth Pro on a sample image."""

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load TensorRT engine.
    engine = load_engine(str(args.trt_model_path))
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

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
                inputs[0]['host'] = input_tensor
                trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
                prediction = {
                    "depth": torch.tensor(trt_outputs[0]),
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
                inputs[0]['host'] = input_tensor
                # 开始计时
                starter.record(stream=torch.cuda.current_stream())
                trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
                # 结束计时
                ender.record(stream=torch.cuda.current_stream())
                prediction = {
                    "depth": torch.tensor(trt_outputs[0]),
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

                    inverse_depth = 1 / depth
                    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
                    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
                    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
                    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                        max_invdepth_vizu - min_invdepth_vizu
                    )

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
    per_image_time = avg_epoch_time / 50.0
    print(
        f"average inference time: {per_image_time} ms, tested on {args.all_epochs} epochs with {args.warm_up_epochs} warm up epochs."
    )

    LOGGER.info("Done predicting depth!")


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with TensorRT models."
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
        default="./output/tensorrt/img1",
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
        "--trt-model-path",
        type=Path,
        default="./checkpoints/tensorrt/model.trt",
        help="Path to the TensorRT model file.",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
