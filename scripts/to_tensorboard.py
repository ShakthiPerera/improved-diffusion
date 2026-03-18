"""
Write per-model reverse-path analysis results to TensorBoard.

Reads .npz stat files and the PRDC CSV, logs everything as TensorBoard scalars
and eigenvalue histograms. Each model gets its own sub-directory so all models
can be compared in a single TensorBoard session.

Usage (single model):
  python scripts/to_tensorboard.py \
      --model_dir reverse_path_analysis/M0_ddpm_eps \
      --model_id  M0_ddpm_eps \
      --tb_dir    reverse_path_analysis/tensorboard

Usage (all models at once):
  for M in M0_ddpm_eps M1_ddpm_x0 M2_endiff_eps M3_endiff_x0 M4_perdim M5_fullcov; do
    python scripts/to_tensorboard.py --model_dir reverse_path_analysis/${M} --model_id ${M}
  done

View in TensorBoard:
  tensorboard --logdir reverse_path_analysis/tensorboard
"""

import argparse
import glob
import os
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
    parser.add_argument('--tb_dir',    type=str,
                        default='reverse_path_analysis/tensorboard')
    args = parser.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    log_dir = os.path.join(args.tb_dir, args.model_id)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Writing TensorBoard logs to {log_dir}")

    # ── Per-timestep stats from .npz files ───────────────────────────────
    npz_files = sorted(glob.glob(os.path.join(args.model_dir, 'stats', 't*.npz')))
    print(f"Processing {len(npz_files)} timestep files ...")

    for npz_file in npz_files:
        t = int(os.path.basename(npz_file).replace('t', '').replace('.npz', ''))
        d = np.load(npz_file)
        eig_cov = d['eigenvalues_cov']
        eig_sec = d['eigenvalues_second']

        # r(t) decomposition
        writer.add_scalar('r_t/value',              float(d['r_t']),               t)
        writer.add_scalar('r_t/trace_cov_over_d',   float(d['trace_cov_over_d']),  t)
        writer.add_scalar('r_t/mean_norm_sq_over_d',float(d['mean_norm_sq_over_d']),t)

        # Covariance off-diagonal sum (0 = diagonal/identity, nonzero = correlation)
        writer.add_scalar('covariance/off_diag_sum', float(d['off_diag_sum_cov']), t)

        # Per-sample r_t
        writer.add_scalar('r_t_per_sample/mean', float(d['r_t_per_sample_mean']), t)
        writer.add_scalar('r_t_per_sample/std',  float(d['r_t_per_sample_std']),  t)
        writer.add_scalar('r_t_per_sample/q25',  float(d['r_t_per_sample_q25']),  t)
        writer.add_scalar('r_t_per_sample/q75',  float(d['r_t_per_sample_q75']),  t)

        # Step dynamics
        writer.add_scalar('step_dynamics/delta_mag_mean',  float(d['delta_mag_mean']),  t)
        writer.add_scalar('step_dynamics/det_mag_mean',    float(d['det_mag_mean']),    t)
        writer.add_scalar('step_dynamics/stoch_mag_mean',  float(d['stoch_mag_mean']),  t)
        writer.add_scalar('step_dynamics/det_stoch_ratio', float(d['det_stoch_ratio']), t)

        # Eigenvalue summary scalars
        writer.add_scalar('eigenvalues/max',              float(eig_cov[0]),                          t)
        writer.add_scalar('eigenvalues/min',              float(eig_cov[-1]),                         t)
        writer.add_scalar('eigenvalues/median',           float(np.median(eig_cov)),                  t)
        writer.add_scalar('eigenvalues/mean',             float(np.mean(eig_cov)),                    t)
        writer.add_scalar('eigenvalues/std',              float(np.std(eig_cov)),                     t)
        writer.add_scalar('eigenvalues/condition_number', float(eig_cov[0] / (eig_cov[-1] + 1e-10)), t)
        writer.add_scalar('eigenvalues/gini',             gini_coefficient(eig_cov),                  t)
        writer.add_scalar('eigenvalues/frac_below_0.9',  float((eig_cov < 0.9).mean()),              t)
        writer.add_scalar('eigenvalues/frac_below_0.5',  float((eig_cov < 0.5).mean()),              t)
        writer.add_scalar('eigenvalues/frac_above_1.1',  float((eig_cov > 1.1).mean()),              t)

        # Eigenvalue distribution as histogram (scrub timestep slider to see evolution)
        writer.add_histogram('eigenspectrum/covariance',     eig_cov, global_step=t)
        writer.add_histogram('eigenspectrum/second_moment',  eig_sec, global_step=t)

    # ── PRDC from image metrics CSV ───────────────────────────────────────
    img_csv = os.path.join(args.model_dir, f'{args.model_id}_image_metrics.csv')
    if os.path.exists(img_csv):
        df = pd.read_csv(img_csv)
        print(f"Writing PRDC for {len(df)} timesteps ...")
        for _, row in df.iterrows():
            t = int(row['t'])
            for metric in ['precision', 'recall', 'density', 'coverage']:
                val = row.get(metric, float('nan'))
                if not np.isnan(val):
                    writer.add_scalar(f'prdc/{metric}', float(val), t)
    else:
        print(f"  [skip] {img_csv} not found")

    writer.close()
    print(f"Done. View with: tensorboard --logdir {args.tb_dir}")


if __name__ == '__main__':
    main()
