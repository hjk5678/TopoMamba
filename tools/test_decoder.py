import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.decoder.decoder import EncoderAlignedDecoder
from utils.losses import TopoMambaLoss


def assert_output(outputs, seg_shape, edge_shapes):
    assert outputs["seg_logits"].shape == seg_shape
    assert len(outputs["edge_preds"]) == len(edge_shapes)
    for edge, shape in zip(outputs["edge_preds"], edge_shapes):
        assert edge.shape == shape, (edge.shape, shape)
        assert torch.isfinite(edge).all()
    assert torch.isfinite(outputs["seg_logits"]).all()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    batch = 2
    out_size = (128, 160)

    decoder = EncoderAlignedDecoder(
        encoder_channels=[64, 128, 256, 512],
        num_classes=6,
    ).to(device).train()

    outputs = decoder(
        torch.randn(batch, 512, 4, 5, device=device),
        torch.randn(batch, 256, 8, 10, device=device),
        torch.randn(batch, 128, 16, 20, device=device),
        torch.randn(batch, 64, 32, 40, device=device),
        out_size=out_size,
    )
    assert_output(
        outputs,
        seg_shape=(batch, 6, *out_size),
        edge_shapes=[
            (batch, 1, 4, 5),
            (batch, 1, 8, 10),
            (batch, 1, 16, 20),
            (batch, 1, 32, 40),
        ],
    )

    target = torch.randint(0, 6, (batch, *out_size), device=device)
    criterion = TopoMambaLoss(
        num_classes=6,
        lambda_edge=0.1,
        lambda_conn=0.0,
    ).to(device)
    loss = criterion(
        outputs["seg_logits"],
        target,
        edge_preds=outputs["edge_preds"],
    )["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()

    missing_grads = [
        name
        for name, parameter in decoder.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    bad_grads = [
        name
        for name, parameter in decoder.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not missing_grads, f"Parameters without gradients: {missing_grads[:20]}"
    assert not bad_grads, f"Non-finite gradients: {bad_grads[:20]}"

    print(f"device: {device}")
    print("Classic U-Net-4 BAU decoder forward/backward smoke test passed")


if __name__ == "__main__":
    main()
