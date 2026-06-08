import argparse
import os
import sys
import time
import types

import torch


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

    def forward(self, x):
        x, (height, width) = self.encoder.prepare_tokens_with_masks(x)
        rope = self.encoder.rope_embed(H=height, W=width)
        for block in self.encoder.blocks:
            x = block._forward(x, rope=rope)
        x = self.encoder.norm(x)
        x = x[:, 5:, :]
        return x.reshape(1, self.grid, self.grid, 1280).permute(0, 3, 1, 2).contiguous()


def rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x, sin, cos):
    return (x * cos) + (rope_rotate_half(x) * sin)


def patch_encoder_for_fixed_shape(encoder, size):
    """Remove trace-hostile dynamic shape plumbing for fixed [1,3,size,size]."""
    grid = size // 16
    token_len = grid * grid + 5

    def patch_embed_forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(1, grid, grid, self.embed_dim)
        return x

    def prepare_tokens_with_masks(self, x, masks=None):
        x = self.patch_embed(x)
        x = x.flatten(1, 2)
        # Fixed B=1 export: avoid aten::Int/expand shape conversion in CoreML.
        return torch.cat([self.cls_token, self.storage_tokens, x], dim=1), (grid, grid)

    encoder.patch_embed.forward = types.MethodType(patch_embed_forward, encoder.patch_embed)
    encoder.prepare_tokens_with_masks = types.MethodType(prepare_tokens_with_masks, encoder)

    def apply_rope_fixed(self, q, k, rope):
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        q_prefix = q[:, :, :5, :]
        k_prefix = k[:, :, :5, :]
        q = torch.cat((q_prefix, rope_apply(q[:, :, 5:, :], sin, cos)), dim=-2)
        k = torch.cat((k_prefix, rope_apply(k[:, :, 5:, :], sin, cos)), dim=-2)
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def compute_attention_fixed(self, qkv, attn_bias=None, rope=None):
        qkv = qkv.reshape(1, token_len, 3, self.num_heads, 64)
        q, k, v = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape(1, token_len, 1280)

    for module in encoder.modules():
        if hasattr(module, "qkv") and hasattr(module, "compute_attention"):
            module.apply_rope = types.MethodType(apply_rope_fixed, module)
            module.compute_attention = types.MethodType(compute_attention_fixed, module)


def load_backbone(checkpoint_dir: str, size: int):
    from sam_3d_body.build_models import load_sam_3d_body

    ckpt = os.path.join(checkpoint_dir, "model.ckpt")
    mhr = os.path.join(checkpoint_dir, "assets", "mhr_model.pt")
    model, _ = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr, device="cpu")
    encoder = model.backbone.encoder.eval().float()
    patch_encoder_for_fixed_shape(encoder, size)
    return BackboneWrapper(encoder, size).eval()


def smoke_forward(wrapper, device: str, size: int):
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


def convert_coreml(wrapper, out_path: str, size: int):
    import coremltools as ct

    example = torch.randn(1, 3, size, size, dtype=torch.float32)
    print("tracing...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
    print("traced %.2fs" % (time.time() - t0), flush=True)
    int_nodes = [node for node in traced.inlined_graph.nodes() if node.kind() == "aten::Int"]
    print("aten::Int nodes after trace:", len(int_nodes), flush=True)
    for node in int_nodes[:20]:
        print("  ", node, flush=True)

    print("converting to CoreML MLProgram fp16...", flush=True)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example.shape)],
        outputs=[ct.TensorType(name="features")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    print("converted %.2fs" % (time.time() - t0), flush=True)

    print("saving", out_path, flush=True)
    mlmodel.save(out_path)
    print("saved", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir",
        default=os.path.join(ROOT, "checkpoints", "sam-3d-body-dinov3"),
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "backbone_coreml.mlpackage"))
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--size", type=int, default=512, help="Input resolution (e.g. 512, 1024)")
    args = ap.parse_args()

    wrapper = load_backbone(args.checkpoint_dir, args.size)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = smoke_forward(wrapper, device, args.size)
    if not args.smoke_only:
        convert_coreml(wrapper, args.out, args.size)


if __name__ == "__main__":
    main()
