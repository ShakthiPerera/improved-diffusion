"""
Convert a samples .npz file (output of image_sample.py) to PNG images.

Saves:
  - A grid image showing the first --grid_n x --grid_n samples
  - Individual PNGs for the first --save_n samples (optional)

Usage:
  python scripts/npz_to_images.py \
      --npz_path logs/samples/M3_endiff_x0/samples_1000x32x32x3.npz \
      --out_dir  logs/samples/M3_endiff_x0/images

  # Just the grid, no individual files:
  python scripts/npz_to_images.py --npz_path <path> --out_dir <dir> --save_n 0
"""

import argparse
import os
import numpy as np
from PIL import Image


def make_grid(images, n_cols):
    """
    images : [N, H, W, 3] uint8 numpy array
    returns: PIL Image of the grid
    """
    N, H, W, C = images.shape
    n_rows = (N + n_cols - 1) // n_cols
    grid = np.zeros((n_rows * H, n_cols * W, C), dtype=np.uint8)
    for idx in range(N):
        r, c = divmod(idx, n_cols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = images[idx]
    return Image.fromarray(grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to the .npz file from image_sample.py')
    parser.add_argument('--out_dir',  type=str, default=None,
                        help='Output directory (default: same folder as npz)')
    parser.add_argument('--grid_n',   type=int, default=10,
                        help='Grid is grid_n x grid_n samples (default: 10)')
    parser.add_argument('--save_n',   type=int, default=0,
                        help='Also save this many individual PNGs (0 = grid only)')
    args = parser.parse_args()

    # Default output dir: same folder as the npz
    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.npz_path))
    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    data = np.load(args.npz_path)
    images = data['arr_0']  # [N, H, W, 3] uint8
    N = len(images)
    print(f"Loaded {N} images  shape={images.shape}  dtype={images.dtype}")

    # Grid
    n_grid = min(args.grid_n * args.grid_n, N)
    grid_img = make_grid(images[:n_grid], n_cols=args.grid_n)
    grid_path = os.path.join(args.out_dir, 'grid.png')
    grid_img.save(grid_path)
    print(f"Saved grid ({args.grid_n}x{args.grid_n}) -> {grid_path}")

    # Individual files
    if args.save_n > 0:
        ind_dir = os.path.join(args.out_dir, 'individual')
        os.makedirs(ind_dir, exist_ok=True)
        n_save = min(args.save_n, N)
        for i in range(n_save):
            Image.fromarray(images[i]).save(os.path.join(ind_dir, f'{i:04d}.png'))
        print(f"Saved {n_save} individual PNGs -> {ind_dir}/")


if __name__ == '__main__':
    main()
