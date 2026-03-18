"""
Reverse-path analysis: run the DDPM reverse chain over N shared samples,
collecting statistics at 25 fixed timesteps.

Per-timestep outputs (saved under output_dir):
  images/t{TTTT}.pt   : [N, 3, 32, 32] fp16 snapshot of x_t
  stats/t{TTTT}.npz   : eigenspectrum, r(t), step dynamics, off-diag covariance sum

Uses the same model loading / sampling methods as scripts/image_sample.py.
x_t is kept on CPU throughout; only batch_size samples move to GPU per forward pass.

Usage (example):
  CUDA_VISIBLE_DEVICES=0 python scripts/reverse_path_analysis.py \
      --model_path logs/exp01/ddpm_eps/ema_0.9999_150000.pt \
      --output_dir reverse_path_analysis/M0_ddpm_eps \
      --z_T_path reverse_path_analysis/shared/z_T.pt \
      --num_samples 10000 \
      --batch_size 128 \
      --image_size 32 --num_channels 128 --num_res_blocks 3 \
      --learn_sigma False --dropout 0.1 --attention_resolutions 16,8 \
      --use_scale_shift_norm True \
      --diffusion_steps 1000 --noise_schedule linear --rescale_timesteps True
"""

import argparse
import os
import time
import numpy as np
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# Timesteps (1-indexed noise level) at which to collect stats.
COLLECT_TIMESTEPS = set(
    [1000, 900, 800, 700, 600, 500]
    + [450, 400, 350, 300, 250, 200, 150]
    + [130, 110, 90, 70, 50, 30]
    + [25, 20, 15, 10, 5]
    + [0]  # final denoised images — handled after the loop
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def gini_coefficient(values):
    v = np.sort(np.abs(values).astype(np.float64))
    n = len(v)
    if v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def collect_eps_stats(diffusion, model, x_t, i, device, batch_size, clip_denoised):
    """
    Run p_mean_variance over all samples in chunks, collect eps predictions.
    Returns eps_flat [N, d] fp32 CPU.
    """
    all_eps = []
    N = x_t.shape[0]
    for s in range(0, N, batch_size):
        chunk = x_t[s:s + batch_size].float().to(device)
        t_batch = th.tensor([i] * chunk.shape[0], device=device)
        with th.no_grad():
            pout = diffusion.p_mean_variance(
                model, chunk, t_batch, clip_denoised=clip_denoised
            )
            eps = diffusion._predict_eps_from_xstart(
                chunk, t_batch, pout['pred_xstart']
            )
        all_eps.append(eps.view(chunk.shape[0], -1).float().cpu())
        del chunk, pout, eps
    return th.cat(all_eps)  # [N, d]


def take_reverse_step(diffusion, model, x_t, i, device, batch_size,
                      clip_denoised, collect_dynamics):
    """
    Run p_mean_variance + manual noise sampling (mirrors p_sample) over all
    samples in chunks.  If collect_dynamics=True, also returns det/stoch split.
    Returns x_next [N, C, H, W] fp16 CPU and dynamics dict (or None).
    """
    N = x_t.shape[0]
    x_next_cpu = th.empty_like(x_t)   # fp16, CPU

    if collect_dynamics:
        all_delta, all_det, all_stoch = [], [], []

    for s in range(0, N, batch_size):
        chunk = x_t[s:s + batch_size].float().to(device)
        t_batch = th.tensor([i] * chunk.shape[0], device=device)
        with th.no_grad():
            pout = diffusion.p_mean_variance(
                model, chunk, t_batch, clip_denoised=clip_denoised
            )
        # Replicate p_sample's noise sampling (3 lines)
        noise = th.randn_like(chunk)
        nonzero = (t_batch != 0).float().view(-1, 1, 1, 1)
        stoch = nonzero * th.exp(0.5 * pout['log_variance']) * noise
        x_new = pout['mean'] + stoch

        if collect_dynamics:
            all_delta.append((x_new - chunk).view(chunk.shape[0], -1).square().mean(1).cpu())
            all_det.append((pout['mean'] - chunk).view(chunk.shape[0], -1).square().mean(1).cpu())
            all_stoch.append(stoch.view(chunk.shape[0], -1).square().mean(1).cpu())

        x_next_cpu[s:s + batch_size] = x_new.half().cpu()
        del chunk, pout, noise, stoch, x_new

    dynamics = None
    if collect_dynamics:
        dynamics = {
            'delta_mag_mean': float(th.cat(all_delta).mean()),
            'det_mag_mean':   float(th.cat(all_det).mean()),
            'stoch_mag_mean': float(th.cat(all_stoch).mean()),
        }
    return x_next_cpu, dynamics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'stats'),  exist_ok=True)

    print("Loading model ...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location='cpu')
    )
    model.to(dist_util.dev())
    model.eval()
    device = dist_util.dev()
    print(f"Model on {device}")

    # Load shared z_T
    print(f"Loading z_T from {args.z_T_path} ...")
    x_t = th.load(args.z_T_path).half()[:args.num_samples]  # [N, 3, 32, 32] fp16 CPU
    N = x_t.shape[0]
    B = args.batch_size
    print(f"Running reverse chain: N={N}, batch_size={B}, T={diffusion.num_timesteps}")

    total_start = time.time()

    for i in range(diffusion.num_timesteps - 1, -1, -1):
        noisy_level = i + 1   # 1-indexed noise level of current x_t

        is_collect = noisy_level in COLLECT_TIMESTEPS

        if is_collect:
            t0 = time.time()

            # ── Collect eps stats ──────────────────────────────────────────
            eps_flat = collect_eps_stats(
                diffusion, model, x_t, i, device, B, args.clip_denoised
            )
            d = eps_flat.shape[1]

            # Save current x_t snapshot
            th.save(x_t.half(), os.path.join(args.output_dir, 'images', f't{noisy_level:04d}.pt'))

            # r(t) decomposition
            r_t = float(eps_flat.square().mean())
            eps_mean = eps_flat.mean(dim=0)                          # [d]
            eps_centered = eps_flat - eps_mean.unsqueeze(0)
            trace_cov_over_d = float(eps_centered.square().mean())
            mean_norm_sq_over_d = float(eps_mean.square().sum() / d)

            # Off-diagonal covariance sum (avoids constructing full d×d matrix)
            # cov_sum - trace(cov) = [sum(row_sums^2) - ||eps_centered||_F^2] / (N-1)
            row_sums = eps_centered.sum(dim=1)                       # [N]
            off_diag_sum_cov = float(
                (row_sums.square().sum() - eps_centered.square().sum()) / (N - 1)
            )

            # Per-sample r_t
            r_t_per = eps_flat.square().mean(dim=1)

            # Eigenspectrum: singular values only (no eigenvectors needed)
            svd_start = time.time()
            S_cov = th.linalg.svdvals(eps_centered)
            eigenvalues_cov = (S_cov ** 2 / (N - 1)).numpy()        # [min(N,d)]

            S_sec = th.linalg.svdvals(eps_flat)
            eigenvalues_second = (S_sec ** 2 / N).numpy()            # [min(N,d)]
            svd_time = time.time() - svd_start

        # ── Reverse step (every timestep) ─────────────────────────────────
        x_t, dynamics = take_reverse_step(
            diffusion, model, x_t, i, device, B,
            args.clip_denoised,
            collect_dynamics=is_collect
        )

        if is_collect:
            step_time = time.time() - t0

            np.savez_compressed(
                os.path.join(args.output_dir, 'stats', f't{noisy_level:04d}.npz'),
                # Eigenspectrum
                eigenvalues_cov=eigenvalues_cov,
                eigenvalues_second=eigenvalues_second,
                # r(t) decomposition
                r_t=r_t,
                trace_cov_over_d=trace_cov_over_d,
                mean_norm_sq_over_d=mean_norm_sq_over_d,
                off_diag_sum_cov=off_diag_sum_cov,
                # Per-sample r_t
                r_t_per_sample_mean=float(r_t_per.mean()),
                r_t_per_sample_std=float(r_t_per.std()),
                r_t_per_sample_q25=float(r_t_per.quantile(0.25)),
                r_t_per_sample_q75=float(r_t_per.quantile(0.75)),
                # Step dynamics
                delta_mag_mean=dynamics['delta_mag_mean'],
                det_mag_mean=dynamics['det_mag_mean'],
                stoch_mag_mean=dynamics['stoch_mag_mean'],
                det_stoch_ratio=dynamics['det_mag_mean'] / (dynamics['stoch_mag_mean'] + 1e-10),
            )

            print(
                f"  t={noisy_level:4d} | r(t)={r_t:.4f} | off_diag={off_diag_sum_cov:.2f} | "
                f"svd={svd_time:.1f}s | step={step_time:.1f}s"
            )

    # ── Save final x_0 images if requested ───────────────────────────────
    if 0 in COLLECT_TIMESTEPS:
        th.save(x_t.half(), os.path.join(args.output_dir, 'images', 't0000.pt'))
        print("Saved final images at t=0")

    total_time = time.time() - total_start
    print(f"\nDone. Total time: {total_time/60:.1f} min")
    print(f"Outputs: {args.output_dir}/images/ and {args.output_dir}/stats/")


def create_argparser():
    defaults = dict(
        model_path='',
        output_dir='',
        z_T_path='reverse_path_analysis/shared/z_T.pt',
        num_samples=10000,
        batch_size=128,
        clip_denoised=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()
