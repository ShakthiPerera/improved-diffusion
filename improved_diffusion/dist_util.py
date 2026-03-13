"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from the environment
    (set automatically by torchrun / torch.distributed.launch for multi-GPU).
    Falls back to single-process defaults if env vars are not set.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    # Defaults for single-GPU / plain `python` launch
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for the current rank.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{dist.get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    Rank 0 reads the file; the bytes are broadcast to all other ranks.
    """
    if dist.get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data_list = [data]
    dist.broadcast_object_list(data_list, src=0)
    return th.load(io.BytesIO(data_list[0]), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
