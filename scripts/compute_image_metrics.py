"""
Compute PRDC (Precision, Recall, Density, Coverage) from saved x_t snapshots.

Reads:  {model_dir}/images/t*.pt
Writes: {model_dir}/{model_id}_image_metrics.csv

Only runs for t <= prdc_max_t (images at high t look like noise; PRDC is meaningless there).

Usage:
  python scripts/compute_image_metrics.py \
      --model_dir reverse_path_analysis/M0_ddpm_eps \
      --model_id M0_ddpm_eps \
      --real_features_path reverse_path_analysis/shared/real_inception_features.pt \
      --prdc_max_t 200
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def get_inception_model(device):
    import torchvision.models as models
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.aux_logits = False
    inception.eval()
    inception.to(device)
    return inception


def extract_inception_features(images, device, batch_size=64):
    """images: [N, 3, H, W] in [-1, 1]. Returns [N, 2048] CPU tensor."""
    inception = get_inception_model(device)
    features_list = []

    def _hook(module, inp, out):
        features_list.append(out.squeeze(-1).squeeze(-1).detach().cpu())

    hook = inception.avgpool.register_forward_hook(_hook)
    for s in range(0, len(images), batch_size):
        chunk = images[s:s + batch_size].float()
        chunk = (chunk + 1.0) / 2.0
        chunk = F.interpolate(chunk, size=(299, 299), mode='bilinear', align_corners=False)
        chunk = chunk.to(device)
        with torch.no_grad():
            inception(chunk)
    hook.remove()
    return torch.cat(features_list)


def compute_prdc(real_feats, gen_feats, k=5, max_samples=5000):
    """Naeem et al. 2020: Precision, Recall, Density, Coverage."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.RandomState(42)
    real_feats = np.asarray(real_feats, dtype=np.float32)
    gen_feats  = np.asarray(gen_feats,  dtype=np.float32)

    if len(real_feats) > max_samples:
        real_feats = real_feats[rng.choice(len(real_feats), max_samples, replace=False)]
    if len(gen_feats) > max_samples:
        gen_feats = gen_feats[rng.choice(len(gen_feats), max_samples, replace=False)]

    knn_r = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(real_feats)
    real_radii = knn_r.kneighbors(real_feats)[0][:, k]

    knn_g = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(gen_feats)
    gen_radii = knn_g.kneighbors(gen_feats)[0][:, k]

    # Precision
    d_gr, i_gr = knn_r.kneighbors(gen_feats, n_neighbors=1)
    precision = float((d_gr[:, 0] <= real_radii[i_gr[:, 0]]).mean())

    # Recall
    d_rg, i_rg = knn_g.kneighbors(real_feats, n_neighbors=1)
    recall = float((d_rg[:, 0] <= gen_radii[i_rg[:, 0]]).mean())

    # Density: (1/(k*Ng)) * sum over gen of (# real in sphere)
    d_gr_k, i_gr_k = knn_r.kneighbors(gen_feats, n_neighbors=k)
    density = float((d_gr_k <= real_radii[i_gr_k]).sum()) / (k * len(gen_feats))

    # Coverage: fraction of real covered by at least one gen sphere
    d_rg_k, i_rg_k = knn_g.kneighbors(real_feats, n_neighbors=k)
    coverage = float((d_rg_k <= gen_radii[i_rg_k]).any(axis=1).mean())

    return {'precision': precision, 'recall': recall,
            'density': density, 'coverage': coverage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',          type=str, required=True)
    parser.add_argument('--model_id',           type=str, required=True)
    parser.add_argument('--real_features_path', type=str,
                        default='reverse_path_analysis/shared/real_inception_features.pt')
    parser.add_argument('--prdc_max_t',         type=int, default=200)
    parser.add_argument('--batch_size',         type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    real_features = torch.load(args.real_features_path).numpy()
    print(f"Loaded real features: {real_features.shape}")

    image_files = sorted(glob.glob(os.path.join(args.model_dir, 'images', 't*.pt')))
    print(f"Found {len(image_files)} image files; running PRDC for t <= {args.prdc_max_t}")

    rows = []
    for pt_file in image_files:
        t = int(os.path.basename(pt_file).replace('t', '').replace('.pt', ''))
        if t > args.prdc_max_t:
            continue
        print(f"  t={t} ...")
        images = torch.load(pt_file).float()
        gen_feats = extract_inception_features(images, device, batch_size=args.batch_size).numpy()
        prdc = compute_prdc(real_features, gen_feats)
        rows.append({'t': t, **prdc})

    out_csv = os.path.join(args.model_dir, f'{args.model_id}_image_metrics.csv')
    pd.DataFrame(rows).sort_values('t', ascending=False).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}  ({len(rows)} rows)")


if __name__ == '__main__':
    main()
