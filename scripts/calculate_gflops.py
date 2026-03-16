import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from improved_diffusion.script_util import create_model

B = 128
model = create_model(
    image_size=32, num_channels=128, num_res_blocks=3,
    learn_sigma=False, class_cond=False, use_checkpoint=False,
    attention_resolutions="16,8", num_heads=4, num_heads_upsample=-1,
    use_scale_shift_norm=True, dropout=0.1,
).eval()

dummy_x = torch.randn(B, 3, 32, 32)
dummy_t = torch.randint(0, 1000, (B,))
flops = FlopCountAnalysis(model, (dummy_x, dummy_t))
flops.unsupported_ops_warnings(False)
unet_gflops = flops.total() / 1e9


class MseLoss(nn.Module):
    def forward(self, pred, target):
        return (pred - target).pow(2).mean(dim=list(range(1, pred.ndim))).mean()

class BatchMeanPenalty(nn.Module):
    # Strategy 02: (1 - E[||ε̂||²/d])² — gaussian_diffusion.py:787-789
    def forward(self, eps_pred):
        r_t_batch = (eps_pred ** 2).view(eps_pred.shape[0], -1).mean()
        return (1.0 - r_t_batch) ** 2

class PerDimPenalty(nn.Module):
    # Strategy 03: per-dim variance — gaussian_diffusion.py:791-794
    def forward(self, eps_pred):
        eps_flat = eps_pred.view(eps_pred.shape[0], -1)
        var_per_dim = eps_flat.square().mean(dim=0)
        return ((var_per_dim - 1.0) ** 2).mean()

class FullCovPenalty(nn.Module):
    # Strategy 04: full covariance — gaussian_diffusion.py:796-801
    def forward(self, eps_pred):
        B = eps_pred.shape[0]
        eps_flat = eps_pred.view(B, -1)
        cov = (eps_flat.T @ eps_flat) / B
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        return ((cov - eye) ** 2).mean()


dummy_eps = torch.randn(B, 3, 32, 32)

def measure_gflops(module, *inputs):
    flops = FlopCountAnalysis(module, inputs)
    flops.unsupported_ops_warnings(False)
    return flops.total() / 1e9

mse_gflops        = measure_gflops(MseLoss(), dummy_eps, dummy_eps)
batch_mean_gflops = measure_gflops(BatchMeanPenalty(), dummy_eps)
per_dim_gflops    = measure_gflops(PerDimPenalty(), dummy_eps)
full_cov_gflops   = measure_gflops(FullCovPenalty(), dummy_eps)

experiments = [
    ("Normal DDPM",          0.0),
    ("EnDiff Strat 02 (batch_mean)", batch_mean_gflops),
    ("EnDiff Strat 03 (per_dim)",    per_dim_gflops),
    ("EnDiff Strat 04 (full_cov)",   full_cov_gflops),
]

# ── Summary Table ────────────────────────────────────────────────────────────
print(f"\nBatch size: {B}  |  Image: 32×32×3  |  d = {3*32*32}")
print(f"Backward ≈ 2× (Fwd+Loss)  →  TrainStep = 3× (Fwd+Loss)\n")

col = "{:<32} {:>8} {:>8} {:>10} {:>12} {:>12} {:>14}"
sep = "-" * 100
print(col.format("Experiment", "UNet(GF)", "MSE(GF)", "Penalty(GF)", "Fwd+Loss(GF)", "Backward(GF)", "TrainStep(GF)"))
print(sep)
for name, penalty in experiments:
    fwd_loss   = unet_gflops + mse_gflops + penalty
    backward   = 2.0 * fwd_loss
    train_step = fwd_loss + backward          # = 3 × fwd_loss
    print(col.format(name, f"{unet_gflops:.3f}", f"{mse_gflops:.3f}",
                     f"{penalty:.4f}", f"{fwd_loss:.3f}", f"{backward:.3f}", f"{train_step:.3f}"))
print(sep)
