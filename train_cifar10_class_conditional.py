"""
Train a class-conditional diffusion model on CIFAR-10.

This is a CIFAR-10-focused wrapper around the standard image training entry
point. It enables class conditioning by default and uses 10 label embeddings
instead of the ImageNet-oriented default.
"""

import argparse
import importlib.util
import os
from pathlib import Path

from improved_diffusion import dist_util, logger
from improved_diffusion import script_util
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from improved_diffusion.train_util import TrainLoop


CIFAR10_NUM_CLASSES = 10


def main():
    args = create_argparser().parse_args()

    if args.prepare_data:
        prepare_cifar10_data(Path(__file__).resolve().parent / "datasets" / "cifar10.py")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Run with --prepare_data True or create it with `python3 datasets/cifar10.py`."
        )

    os.environ.setdefault(
        "OPENAI_LOGDIR",
        str(Path(__file__).resolve().parent / "logs" / "cifar10_class_conditional"),
    )

    script_util.NUM_CLASSES = args.num_classes

    dist_util.setup_dist()
    logger.configure()

    logger.log(f"using {args.num_classes} classes for class conditioning")
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def prepare_cifar10_data(script_path):
    spec = importlib.util.spec_from_file_location("prepare_cifar10", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def create_argparser():
    repo_dir = Path(__file__).resolve().parent
    defaults = dict(
        data_dir=str(repo_dir / "datasets" / "cifar_train"),
        schedule_sampler="uniform",
        lr=2e-4,
        weight_decay=0.0,
        lr_anneal_steps=800000,
        batch_size=128,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_classes=CIFAR10_NUM_CLASSES,
        prepare_data=False,
    )
    model_defaults = model_and_diffusion_defaults()
    model_defaults.update(
        image_size=32,
        num_channels=128,
        num_res_blocks=3,
        learn_sigma=False,
        dropout=0.1,
        class_cond=True,
        diffusion_steps=1000,
        noise_schedule="linear",
    )
    defaults.update(model_defaults)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
