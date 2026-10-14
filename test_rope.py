import torch
import torch.nn.functional as F
from einops import rearrange

def diffsynth_rope(pos: torch.Tensor, dim: int, theta: float = 1000.0, ntk: float = 1.0) -> torch.Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def diffsynth_posemb(pos, axdims=[32, 48, 48], theta=1000.0):
    return torch.cat(
        [diffsynth_rope(pos[..., i], d, theta, 1.0) for i, d in enumerate(axdims)],
        dim=-3,
    )

def diffsynth_ropeapply(xq: torch.Tensor, freqs: torch.Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype)

def rust_ropeapply(xq: torch.Tensor, pos: torch.Tensor, theta: float = 1000.0):
    # Rust compute_krea_rope:
    axes = [32, 48, 48]
    angles_all = []
    for axis_idx, axis_dim in enumerate(axes):
        half = axis_dim // 2
        half_freqs = []
        for step in range(half):
            scale_val = (2.0 * step) / float(axis_dim)
            omega = 1.0 / (theta ** scale_val)
            half_freqs.append(omega)
        omega_t = torch.tensor(half_freqs, device=pos.device).unsqueeze(0).unsqueeze(0) # [1, 1, half]
        pos_axis = pos[..., axis_idx:axis_idx+1] # [B, L, 1]
        angles = pos_axis * omega_t
        angles_all.append(angles)
    angles_cat = torch.cat(angles_all, dim=2) # [B, L, 64]
    cos = angles_cat.cos().unsqueeze(1).unsqueeze(4) # [B, 1, L, 64, 1]
    sin = angles_cat.sin().unsqueeze(1).unsqueeze(4) # [B, 1, L, 64, 1]

    # Rust apply_rope:
    tb, th, tl, td = xq.shape
    t_f32 = xq.float().reshape(tb, th, tl, td // 2, 2)
    u0 = t_f32[..., 0:1] # [B, H, L, 64, 1]
    u1 = t_f32[..., 1:2] # [B, H, L, 64, 1]
    # [cos, -sin] * [u0, u1]^T -> new_u0 = cos * u0 - sin * u1
    # [sin,  cos] * [u0, u1]^T -> new_u1 = sin * u0 + cos * u1
    new_u0 = cos * u0 - sin * u1
    new_u1 = sin * u0 + cos * u1
    out = torch.stack([new_u0.squeeze(-1), new_u1.squeeze(-1)], dim=-1).reshape(tb, th, tl, td)
    return out

if __name__ == '__main__':
    torch.manual_seed(42)
    b, h, l, d = 1, 48, 10, 128
    q = torch.randn(b, h, l, d)
    pos = torch.randint(0, 50, (b, l, 3)).float()
    
    freqs = diffsynth_posemb(pos)
    q_diffsynth = diffsynth_ropeapply(q, freqs)
    q_rust = rust_ropeapply(q, pos)

    diff = (q_diffsynth - q_rust).abs().max().item()
    print(f"Max absolute diff between DiffSynth and Rust RoPE: {diff:.8f}")
