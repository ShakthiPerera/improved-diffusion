"""
Generate shared assets used by all reverse-path analysis runs:
  - z_T.pt              : shared initial Gaussian noise [N, 3, 32, 32]
  - real_inception_features.pt : Inception pool3 features for real CIFAR-10 images

Usage:
  python scripts/generate_shared_assets.py \
      --num_samples 10000 \
      --image_dir datasets/cifar_train \
      --output_dir reverse_path_analysis/shared \
      --seed 42
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_real_images(image_dir, num_samples):
    """Load CIFAR images from disk, normalize to [-1, 1] (matches training)."""
    files = sorted(os.listdir(image_dir))
    files = [f for f in files if f.endswith('.png')]
    assert len(files) >= num_samples, (
        f"Only {len(files)} images in {image_dir}, need {num_samples}"
    )
    files = files[:num_samples]

    images = []
    for fname in files:
        img = np.array(Image.open(os.path.join(image_dir, fname)).convert('RGB'))
        img = img.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1, 1]
        images.append(torch.from_numpy(img.transpose(2, 0, 1)))  # [3, H, W]

    return torch.stack(images)  # [N, 3, H, W]


def get_inception_model(device):
    import torchvision.models as models
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.aux_logits = False
    inception.eval()
    inception.to(device)
    return inception


def extract_inception_features(images, device, batch_size=64):
    """
    Extract Inception v3 avgpool features.
    images: [N, 3, H, W] float tensor in [-1, 1]
    returns: [N, 2048] float32 CPU tensor
    """
    inception = get_inception_model(device)

    # Hook to capture avgpool output
    features_list = []
    def _hook(module, inp, out):
        features_list.append(out.squeeze(-1).squeeze(-1).detach().cpu())
    hook = inception.avgpool.register_forward_hook(_hook)

    for s in range(0, len(images), batch_size):
        chunk = images[s:s + batch_size].float()
        # [-1, 1] -> [0, 1], resize to 299x299
        chunk = (chunk + 1.0) / 2.0
        chunk = F.interpolate(chunk, size=(299, 299), mode='bilinear', align_corners=False)
        chunk = chunk.to(device)
        with torch.no_grad():
            inception(chunk)

    hook.remove()
    return torch.cat(features_list)  # [N, 2048]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--image_dir', type=str, default='datasets/cifar_train')
    parser.add_argument('--output_dir', type=str, default='reverse_path_analysis/shared')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Shared initial noise z_T
    z_T_path = os.path.join(args.output_dir, 'z_T.pt')
    if os.path.exists(z_T_path):
        print(f"[skip] {z_T_path} already exists")
    else:
        print(f"Generating z_T with seed={args.seed} ...")
        rng = torch.Generator()
        rng.manual_seed(args.seed)
        z_T = torch.randn(args.num_samples, 3, 32, 32, generator=rng)
        torch.save(z_T, z_T_path)
        print(f"Saved {z_T_path}  shape={z_T.shape}  range=[{z_T.min():.2f}, {z_T.max():.2f}]")

    # 2. Real images
    real_path = os.path.join(args.output_dir, 'cifar10_real.pt')
    if os.path.exists(real_path):
        print(f"[skip] {real_path} already exists")
        real_images = torch.load(real_path)
    else:
        print(f"Loading {args.num_samples} real images from {args.image_dir} ...")
        real_images = load_real_images(args.image_dir, args.num_samples)
        torch.save(real_images, real_path)
        print(f"Saved {real_path}  shape={real_images.shape}  range=[{real_images.min():.2f}, {real_images.max():.2f}]")

    # 3. Inception features for real images
    feats_path = os.path.join(args.output_dir, 'real_inception_features.pt')
    if os.path.exists(feats_path):
        print(f"[skip] {feats_path} already exists")
    else:
        print("Extracting Inception features for real images ...")
        feats = extract_inception_features(real_images, device, batch_size=64)
        torch.save(feats, feats_path)
        print(f"Saved {feats_path}  shape={feats.shape}")

    print("Done.")


if __name__ == '__main__':
    main()
