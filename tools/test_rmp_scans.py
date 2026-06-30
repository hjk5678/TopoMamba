import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.encoder.rmtpb import ResidualMultiPathVSS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    module = ResidualMultiPathVSS(
        dim=64,
        num_blocks=1,
        num_paths=4,
        local_window_size=8,
        local_window_shift=True,
        atrous_rate=2,
    ).to(device)
    module.train()

    # Deliberately use a non-square size that is not divisible by 8 or 2.
    x = torch.randn(2, 64, 33, 47, device=device, requires_grad=True)
    out, path_features = module.forward_with_paths(x)

    assert out.shape == x.shape, (out.shape, x.shape)
    assert len(path_features) == 4
    assert all(feat.shape == (2, 16, 33, 47) for feat in path_features)
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(feat).all() for feat in path_features)

    # gamma starts at zero, so the residual output must initially equal x.
    residual_error = (out - x).abs().max().item()
    assert residual_error == 0.0, residual_error

    loss = out.square().mean()
    loss = loss + sum(feat.square().mean() for feat in path_features)
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    bad_grads = [
        name
        for name, param in module.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    assert not bad_grads, f"non-finite gradients: {bad_grads[:20]}"

    print(f"device: {device}")
    print(f"output: {tuple(out.shape)}")
    for name, feat in zip(module.path_names, path_features):
        print(f"{name}: {tuple(feat.shape)}")
    print("forward/backward smoke test passed")


if __name__ == "__main__":
    main()
