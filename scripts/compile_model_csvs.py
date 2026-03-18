"""
Compile per-model stats .npz files and PRDC CSV into 2 CSVs:
  {model_id}_scalars.csv       : one row per timestep, all scalar metrics
  {model_id}_eigenspectra.csv  : one row per (timestep, eigenvalue_idx)

Usage:
  python scripts/compile_model_csvs.py \
      --model_dir reverse_path_analysis/M0_ddpm_eps \
      --model_id M0_ddpm_eps
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd


def gini_coefficient(values):
    v = np.sort(np.abs(values.astype(np.float64)))
    n = len(v)
    if v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--model_id',  type=str, required=True)
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.model_dir, 'stats', 't*.npz')))
    if not npz_files:
        print(f"No .npz files found in {args.model_dir}/stats/")
        return
    print(f"Found {len(npz_files)} .npz files")

    scalar_rows = []
    eigen_rows  = []

    for npz_file in npz_files:
        t = int(os.path.basename(npz_file).replace('t', '').replace('.npz', ''))
        d = np.load(npz_file)

        eig_cov = d['eigenvalues_cov']
        eig_sec = d['eigenvalues_second']

        # ── Scalars ──────────────────────────────────────────────────────
        scalar_rows.append({
            't': t,
            # r(t) decomposition
            'r_t':                  float(d['r_t']),
            'trace_cov_over_d':     float(d['trace_cov_over_d']),
            'mean_norm_sq_over_d':  float(d['mean_norm_sq_over_d']),
            'off_diag_sum_cov':     float(d['off_diag_sum_cov']),
            # Per-sample r_t
            'r_t_per_sample_mean':  float(d['r_t_per_sample_mean']),
            'r_t_per_sample_std':   float(d['r_t_per_sample_std']),
            'r_t_per_sample_q25':   float(d['r_t_per_sample_q25']),
            'r_t_per_sample_q75':   float(d['r_t_per_sample_q75']),
            # Step dynamics
            'delta_mag_mean':       float(d['delta_mag_mean']),
            'det_mag_mean':         float(d['det_mag_mean']),
            'stoch_mag_mean':       float(d['stoch_mag_mean']),
            'det_stoch_ratio':      float(d['det_stoch_ratio']),
            # Eigenvalue summaries
            'eigen_max':            float(eig_cov[0]),
            'eigen_min':            float(eig_cov[-1]),
            'eigen_median':         float(np.median(eig_cov)),
            'eigen_mean':           float(np.mean(eig_cov)),
            'eigen_std':            float(np.std(eig_cov)),
            'condition_number':     float(eig_cov[0] / (eig_cov[-1] + 1e-10)),
            'eigen_gini':           gini_coefficient(eig_cov),
            'frac_eigen_below_0.9': float((eig_cov < 0.9).mean()),
            'frac_eigen_below_0.5': float((eig_cov < 0.5).mean()),
            'frac_eigen_above_1.1': float((eig_cov > 1.1).mean()),
        })

        # ── Eigenspectra (full spectrum) ─────────────────────────────────
        for idx in range(len(eig_cov)):
            eigen_rows.append({
                't': t,
                'eigen_idx':         idx,
                'eigenvalue_cov':    float(eig_cov[idx]),
                'eigenvalue_second': float(eig_sec[idx]),
            })

    # ── Merge with PRDC ──────────────────────────────────────────────────
    img_csv = os.path.join(args.model_dir, f'{args.model_id}_image_metrics.csv')
    stats_df = pd.DataFrame(scalar_rows)

    if os.path.exists(img_csv):
        img_df = pd.read_csv(img_csv)
        merged = stats_df.merge(img_df, on='t', how='outer')
    else:
        print(f"  [warn] {img_csv} not found — scalars CSV will not include PRDC")
        merged = stats_df

    merged = merged.sort_values('t', ascending=False)

    out_scalars = os.path.join(args.model_dir, f'{args.model_id}_scalars.csv')
    out_eigen   = os.path.join(args.model_dir, f'{args.model_id}_eigenspectra.csv')

    merged.to_csv(out_scalars, index=False)
    (pd.DataFrame(eigen_rows)
       .sort_values(['t', 'eigen_idx'], ascending=[False, True])
       .to_csv(out_eigen, index=False))

    print(f"Saved:")
    print(f"  {out_scalars}  ({len(merged)} rows)")
    print(f"  {out_eigen}  ({len(eigen_rows)} rows)")


if __name__ == '__main__':
    main()
