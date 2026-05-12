"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random
import struct
import zlib

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

CIFAR10_CLASS_NAMES = (
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "plane",
    "ship",
    "truck",
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    if args.seed >= 0:
        seed = args.seed + dist.get_rank()
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
    label_generator = None
    if args.label_seed >= 0:
        label_generator = th.Generator(device="cpu")
        label_generator.manual_seed(args.label_seed + dist.get_rank())
    logger.configure()

    balanced_labels = None
    if args.label_mode not in ("random", "balanced"):
        raise ValueError(f"unknown label_mode: {args.label_mode}")
    if args.class_cond and args.label_mode == "balanced":
        if args.num_samples % args.num_classes != 0:
            raise ValueError(
                f"balanced label mode requires num_samples ({args.num_samples}) "
                f"to be divisible by num_classes ({args.num_classes})"
            )
        labels_per_class = args.num_samples // args.num_classes
        balanced_labels = th.arange(args.num_classes).repeat_interleave(labels_per_class)
        if args.label_seed >= 0:
            balanced_generator = th.Generator(device="cpu")
            balanced_generator.manual_seed(args.label_seed)
            perm = th.randperm(len(balanced_labels), generator=balanced_generator)
            balanced_labels = balanced_labels[perm]

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    png_dir = args.png_dir or os.path.join(logger.get_dir(), "png_samples")
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            if balanced_labels is not None:
                label_start = len(all_images) * args.batch_size
                rank_start = label_start + dist.get_rank() * args.batch_size
                classes = balanced_labels[
                    rank_start : rank_start + args.batch_size
                ].to(dist_util.dev())
                if classes.shape[0] < args.batch_size:
                    padding = th.zeros(
                        args.batch_size - classes.shape[0],
                        dtype=balanced_labels.dtype,
                        device=dist_util.dev(),
                    )
                    classes = th.cat([classes, padding], dim=0)
            elif label_generator is None:
                classes = th.randint(
                    low=0,
                    high=args.num_classes,
                    size=(args.batch_size,),
                    device=dist_util.dev(),
                )
            else:
                classes = th.randint(
                    low=0,
                    high=args.num_classes,
                    size=(args.batch_size,),
                    generator=label_generator,
                    device="cpu",
                ).to(dist_util.dev())
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        saved_count = len(all_images) * args.batch_size
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_sample_arrays = [sample.cpu().numpy() for sample in gathered_samples]
        all_images.extend(gathered_sample_arrays)

        gathered_label_arrays = None
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            gathered_label_arrays = [labels.cpu().numpy() for labels in gathered_labels]
            all_labels.extend(gathered_label_arrays)

        if args.save_png and dist.get_rank() == 0:
            batch_arr = np.concatenate(gathered_sample_arrays, axis=0)
            remaining = args.num_samples - saved_count
            batch_arr = batch_arr[:remaining]
            if args.class_cond:
                batch_labels = np.concatenate(gathered_label_arrays, axis=0)[:remaining]
                save_png_samples(
                    batch_arr,
                    png_dir,
                    labels=batch_labels,
                    start_index=saved_count,
                )
            else:
                save_png_samples(batch_arr, png_dir, start_index=saved_count)

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        if args.save_png:
            logger.log(f"png samples saved to {png_dir}")

    dist.barrier()
    logger.log("sampling complete")


def save_png_samples(arr, png_dir, labels=None, start_index=0):
    os.makedirs(png_dir, exist_ok=True)
    for i, image in enumerate(arr):
        sample_index = start_index + i
        if labels is None:
            filename = f"sample_{sample_index:06d}.png"
        else:
            label = int(labels[i])
            class_name = (
                CIFAR10_CLASS_NAMES[label]
                if 0 <= label < len(CIFAR10_CLASS_NAMES)
                else f"class{label}"
            )
            filename = f"{class_name}_{sample_index:06d}.png"
        write_rgb_png(os.path.join(png_dir, filename), image)


def write_rgb_png(path, image):
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {image.shape}")

    height, width, _ = image.shape
    raw_rows = [b"\x00" + np.ascontiguousarray(row).tobytes() for row in image]
    raw = b"".join(raw_rows)

    def chunk(chunk_type, data):
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        seed=-1,
        label_seed=-1,
        label_mode="random",
        save_png=False,
        png_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
