import os
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cluster_graph import MultiScaleClusterGraph


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    module = MultiScaleClusterGraph(
        channels=(64, 128, 256, 512),
        cluster_counts=(64, 32, 16, 8),
        graph_dim=32,
        num_cluster_iters=2,
        spatial_weight=0.5,
    ).to(device).train()

    assert not any(
        isinstance(child, nn.MultiheadAttention)
        for child in module.modules()
    ), "MS-CGC must not contain multi-head attention"

    features = [
        torch.randn(2, 64, 32, 40, device=device),
        torch.randn(2, 128, 16, 20, device=device),
        torch.randn(2, 256, 8, 10, device=device),
        torch.randn(2, 512, 4, 5, device=device),
    ]
    with torch.amp.autocast(
        device_type=device.type,
        enabled=(device.type == "cuda"),
    ):
        outputs = module(features)
        loss = sum(output.square().mean() for output in outputs)

    assert len(outputs) == len(features)
    for output, feature in zip(outputs, features):
        assert output.shape == feature.shape, (output.shape, feature.shape)
        assert torch.isfinite(output).all()

    assert torch.isfinite(loss)
    loss.backward()

    missing_grads = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite_grads = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]

    assert not missing_grads, f"Parameters without gradients: {missing_grads}"
    assert not nonfinite_grads, f"Non-finite gradients: {nonfinite_grads}"

    print(f"device: {device}")
    for index, output in enumerate(outputs, start=1):
        print(f"g{index}: {tuple(output.shape)}")
    print("MS-CGC attention-free forward/backward smoke test passed")


if __name__ == "__main__":
    main()
