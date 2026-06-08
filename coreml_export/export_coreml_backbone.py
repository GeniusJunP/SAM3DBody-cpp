"""Export SAM-3D-Body image backbone (ViT encoder) to CoreML."""

import argparse
import os
import sys
import time

import torch

torch.backends.mha.set_fastpath_enabled(False)

ROOT = os.path.dirname(os.path.abspath(__file__))
FAST_REPO = os.path.join(ROOT, "..", "Fast-SAM-3D-Body")
sys.path.insert(0, FAST_REPO)

os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")


class BackboneWrapper(torch.nn.Module):
    def __init__(self, encoder, size):
        super().__init__()
        self.encoder = encoder
        self.grid = size // 16

        def patched_prepare(x, masks=None):
            x_patched = self.encoder.patch_embed(x)
            B, H, W, _ = x_patched.shape
            x_patched = x_patched.flatten(1, 2)
            
            cls_token = self.encoder.cls_token + 0 * self.encoder.mask_token
            if self.encoder.n_storage_tokens > 0:
                storage_tokens = self.encoder.storage_tokens
            else:
                storage_tokens = torch.empty(1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device)
            
            # Use repeat instead of expand to avoid CoreML BNNS backend std::bad_cast bug
            # with RangeDim / symbolic batch dimension in torch.export.
            x_patched = torch.cat(
                [
                    cls_token.repeat(B, 1, 1),
                    storage_tokens.repeat(B, 1, 1),
                    x_patched,
                ],
                dim=1,
            )
            return x_patched, (H, W)
            
        self.encoder.prepare_tokens_with_masks = patched_prepare

    def forward(self, x):
        b = x.shape[0]
        x, (height, width) = self.encoder.prepare_tokens_with_masks(x)
        rope = self.encoder.rope_embed(H=height, W=width)
        for block in self.encoder.blocks:
            x = block._forward(x, rope=rope)
        x = self.encoder.norm(x)
        x = x[:, 5:, :]
        return x.reshape(b, self.grid, self.grid, 1280).permute(0, 3, 1, 2).contiguous()


def load_backbone(checkpoint_dir: str, size: int) -> BackboneWrapper:
    from sam_3d_body.build_models import load_sam_3d_body

    ckpt = os.path.join(checkpoint_dir, "model.ckpt")
    mhr = os.path.join(checkpoint_dir, "assets", "mhr_model.pt")
    model, _ = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr, device="cpu")
    encoder = model.backbone.encoder.eval().float()
    return BackboneWrapper(encoder, size).eval()


def smoke_forward(wrapper: BackboneWrapper, device: str, size: int) -> BackboneWrapper:
    wrapper = wrapper.to(device)
    x = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        y = wrapper(x)
    if device == "mps":
        torch.mps.synchronize()
    print(
        "forward %.2fs shape=%s dtype=%s mean=%.6f"
        % (time.time() - t0, tuple(y.shape), y.dtype, y.float().mean().item()),
        flush=True,
    )
    return wrapper.cpu()


def convert_coreml(
    wrapper: BackboneWrapper, out_path: str, size: int, verify_only: bool = False
):
    import coremltools as ct

    # Use static B=1 batch size to avoid shape inference issues downstream
    # and Watchdog timeouts. We'll use MLBatchProvider for batching inference.
    example = torch.randn(1, 3, size, size, dtype=torch.float32)
    print("exporting to EXIR...", flush=True)
    t0 = time.time()

    # Use torch.export to preserve the native SDPA op. JIT unrolls it.
    exported = torch.export.export(
        wrapper,
        (example,),
    )
    exported = exported.run_decompositions()
    print("exported %.2fs" % (time.time() - t0), flush=True)

    print("converting to CoreML MLProgram...", flush=True)
    t0 = time.time()
    mlmodel = ct.convert(
        exported,
        inputs=[ct.TensorType(name="image", shape=example.shape)],
        outputs=[ct.TensorType(name="features")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS14,
        skip_model_load=verify_only,
    )
    print("converted %.2fs" % (time.time() - t0), flush=True)

    if verify_only:
        print("CoreML conversion feasible.", flush=True)
        return

    print("saving", out_path, flush=True)
    mlmodel.save(out_path)
    size_mb = (
        sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(out_path)
            for f in fns
        )
        / 1e6
    )
    print("saved %s (%.1f MB)" % (out_path, size_mb), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Export SAM-3D-Body backbone to CoreML via EXIR"
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=os.path.join(ROOT, "checkpoints", "sam-3d-body-dinov3"),
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "backbone_coreml.mlpackage"))
    ap.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run forward pass only, skip conversion",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Convert but skip model load (feasibility check)",
    )
    ap.add_argument(
        "--size", type=int, default=512, help="Input resolution (e.g. 512, 1024)"
    )
    args = ap.parse_args()

    wrapper = load_backbone(args.checkpoint_dir, args.size)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = smoke_forward(wrapper, device, args.size)
    if not args.smoke_only:
        convert_coreml(wrapper, args.out, args.size, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
