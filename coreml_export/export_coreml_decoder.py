"""
Phase 1 verification: Can the Decoder be traced and converted to CoreML?

This script applies monkey-patches to eliminate aten::Int nodes and
tests torch.jit.trace + coremltools.convert feasibility.
"""
import os
import sys
import time
import types

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
FAST_REPO = os.path.join(ROOT, "..", "Fast-SAM-3D-Body")
sys.path.insert(0, FAST_REPO)

os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")

# ── Fixed constants (verified from model) ──────────────────────────────
B = 1

BACKBONE_DIM = 1280
DECODER_DIM = 1024


# ══════════════════════════════════════════════════════════════════════════
# CoreML Decoder Wrapper — all shapes hardcoded, no CUDA dependencies
# ══════════════════════════════════════════════════════════════════════════
class CoreMLDecoderWrapper(nn.Module):
    """
    Wraps CameraEncoder + PromptEncoder + PromptableDecoder for CoreML export.

    Inputs:  features [1,1280,32,32], cond_info [1,3], ray_cond [1,2,32,32]
    Output:  pose_token [1,1024]
    """

    def __init__(self, model, size):
        super().__init__()
        self.feat_hw = size // 16
        self.ray_cond_emb = model.ray_cond_emb
        self.decoder = model.decoder
        self.init_pose = model.init_pose
        self.init_camera = model.init_camera
        self.init_to_token = model.init_to_token_mhr
        self.prev_to_token = model.prev_to_token_mhr
        self.prompt_encoder = model.prompt_encoder
        self.prompt_to_token = model.prompt_to_token

        for attr in ("keypoint_embedding", "keypoint3d_embedding", "hand_box_embedding"):
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

        self.decoder.do_interm_preds = False
        self.decoder.keypoint_token_update = None

    def forward(self, features, cond_info, ray_cond):
        # ── CameraEncoder ─────────────────────────────────────────────
        H = W = self.feat_hw
        rays = ray_cond.permute(0, 2, 3, 1)                          # [1,H,W,2]
        rays = torch.cat([rays, torch.ones_like(rays[..., :1])], -1)  # [1,H,W,3]
        rays_emb = self.ray_cond_emb.camera(pos=rays.reshape(1, H * W, 3))  # [1,H*W,99]
        rays_emb = rays_emb.reshape(1, H, W, -1).permute(0, 3, 1, 2).contiguous()
        z = torch.cat([features, rays_emb], dim=1)
        features = self.ray_cond_emb.norm(self.ray_cond_emb.conv(z))  # [1,1280,32,32]

        # ── Build initial estimate ────────────────────────────────────
        init_pose = self.init_pose.weight.unsqueeze(0)      # [1,1,519]
        init_camera = self.init_camera.weight.unsqueeze(0)   # [1,1,3]
        init_est = torch.cat([init_pose, init_camera], dim=-1)  # [1,1,522]

        # ── Pose token ────────────────────────────────────────────────
        init_input = torch.cat(
            [cond_info.unsqueeze(1), init_est], dim=-1
        )  # [1,1,525]
        token_seq = self.init_to_token(init_input)  # [1,1,1024]

        # ── Previous-estimate token ───────────────────────────────────
        prev_emb = self.prev_to_token(init_est)  # [1,1,1024]

        # ── Dummy keypoint prompt (label=-2 → "no keypoints") ─────────
        kps = torch.zeros(1, 1, 3, dtype=features.dtype, device=features.device)
        kps[:, :, -1] = -2.0
        prompt_emb, _ = self.prompt_encoder(keypoints=kps)  # [1,1,1280]
        prompt_emb = self.prompt_to_token(prompt_emb)  # [1,1,1024]

        # ── Token sequence + augment ──────────────────────────────────
        token_seq = torch.cat([token_seq, prev_emb, prompt_emb], dim=1)  # [1,3,1024]
        tok_aug = torch.zeros_like(token_seq)
        tok_aug[:, 1] = prev_emb[:, 0]
        tok_aug[:, 2] = prompt_emb[:, 0]

        for attr in ("keypoint_embedding", "keypoint3d_embedding", "hand_box_embedding"):
            if hasattr(self, attr):
                emb = getattr(self, attr).weight.unsqueeze(0)  # [1,N,1024]
                token_seq = torch.cat([token_seq, emb], dim=1)
                tok_aug = torch.cat([tok_aug, torch.zeros_like(emb)], dim=1)

        # ── Image positional encoding ─────────────────────────────────
        img_pe = self.prompt_encoder.get_dense_pe((self.feat_hw, self.feat_hw))  # [1,1280,H,W]

        # ── Run decoder ───────────────────────────────────────────────
        out = self.decoder(
            token_embedding=token_seq,
            image_embedding=features,
            token_augment=tok_aug,
            image_augment=img_pe,
            token_mask=None,
            channel_first=True,
            token_to_pose_output_fn=None,
        )
        if isinstance(out, (tuple, list)):
            token_out = out[0]
        else:
            token_out = out

        return token_out[:, 0]  # [1,1024]


# ══════════════════════════════════════════════════════════════════════════
# Monkey-patches for trace-hostile code
# ══════════════════════════════════════════════════════════════════════════
def patch_for_coreml_trace(wrapper, size):
    """Apply fixed-shape patches to eliminate aten::Int nodes."""
    feat_hw = size // 16

    # ── Patch 1: PromptEncoder.forward (bypass _embed_keypoints loop) ──
    def _prompt_encoder_forward_fixed(self, keypoints, boxes=None, masks=None):
        # In CoreMLDecoderWrapper, labels is ALWAYS -2.0 (no keypoints).
        # Bypass the 70-joint loop that confuses the tracer.
        coords = keypoints[:, :, :2]
        point_embedding = self.pe_layer._pe_encoding(coords.to(torch.float))
        # For label=-2: zero out + add invalid_point_embed
        point_embedding = torch.zeros_like(point_embedding)
        point_embedding = point_embedding + self.invalid_point_embed.weight
        # mask: all False for label=-2
        point_mask = torch.zeros(
            point_embedding.shape[0], point_embedding.shape[1],
            dtype=torch.bool, device=point_embedding.device,
        )
        return point_embedding, point_mask

    wrapper.prompt_encoder.forward = types.MethodType(
        _prompt_encoder_forward_fixed, wrapper.prompt_encoder
    )

    # ── Patch 2: PositionEmbeddingRandom.forward (fixed h=w=feat_hw) ────────
    def _pe_forward_fixed(self, _size):
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((feat_hw, feat_hw), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / float(feat_hw)
        x_embed = x_embed / float(feat_hw)
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x feat_hw x feat_hw

    wrapper.prompt_encoder.pe_layer.forward = types.MethodType(
        _pe_forward_fixed, wrapper.prompt_encoder.pe_layer
    )

    # ── Patch 3: _generate_fourier_features (eliminate range(b) loop) ──
    import sam_3d_body.models.modules.camera_embed as cam_mod

    _orig_gen = cam_mod._generate_fourier_features

    def _generate_fourier_features_fixed(pos, num_bands, max_resolution):
        # pos: [1, 1024, 3] — B is always 1
        device = pos.device
        min_freq = 1.0
        import numpy as np
        freq_bands = torch.stack(
            [
                torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
                for res in max_resolution
            ],
            dim=0,
        )
        # Replace loop: pos[0] instead of for i in range(b)
        per_pos_features = (pos[:, :, :, None] * freq_bands[None, None, :, :])
        per_pos_features = per_pos_features.reshape(1, pos.shape[1], -1)
        import numpy as np
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)],
            dim=-1,
        )
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
        return per_pos_features

    cam_mod._generate_fourier_features = _generate_fourier_features_fixed


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    import coremltools as ct

    ap = argparse.ArgumentParser(description="Export SAM-3D-Body decoder to CoreML")
    ap.add_argument(
        "--checkpoint-dir",
        default=os.path.join(ROOT, "checkpoints", "sam-3d-body-dinov3"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "decoder_coreml.mlpackage"),
    )
    ap.add_argument("--verify-only", action="store_true",
                    help="Only verify feasibility, don't save")
    ap.add_argument("--size", type=int, default=512, help="Original backbone size")
    args = ap.parse_args()

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...", flush=True)
    from sam_3d_body.build_models import load_sam_3d_body
    ckpt = os.path.join(args.checkpoint_dir, "model.ckpt")
    mhr = os.path.join(args.checkpoint_dir, "assets", "mhr_model.pt")
    model, _ = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr, device="cpu")

    # ── Build wrapper ─────────────────────────────────────────────────
    print("Building wrapper...", flush=True)
    wrapper = CoreMLDecoderWrapper(model, args.size)
    wrapper.eval()
    wrapper.float()

    print("Applying patches...", flush=True)
    patch_for_coreml_trace(wrapper, args.size)

    # ── Fixed dummy inputs ────────────────────────────────────────────
    feat_hw = args.size // 16
    feat = torch.randn(1, BACKBONE_DIM, feat_hw, feat_hw, dtype=torch.float32)
    cond = torch.randn(1, 3, dtype=torch.float32)
    ray = torch.randn(1, 2, feat_hw, feat_hw, dtype=torch.float32)

    # ── Smoke test ────────────────────────────────────────────────────
    print("Smoke test forward...", flush=True)
    with torch.no_grad():
        ref_out = wrapper(feat, cond, ray)
    print(f"  Output shape: {tuple(ref_out.shape)}, dtype: {ref_out.dtype}", flush=True)

    # ── Trace ─────────────────────────────────────────────────────────
    print("Tracing...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (feat, cond, ray), strict=False)
    print(f"  Traced in {time.time()-t0:.1f}s", flush=True)

    int_nodes = [n for n in traced.inlined_graph.nodes() if n.kind() == "aten::Int"]
    print(f"  aten::Int nodes: {len(int_nodes)}", flush=True)
    for node in int_nodes:
        print(f"    {node}", flush=True)

    # ── Phase 2: Traced model correctness ─────────────────────────────
    print("Phase 2: Checking traced model correctness...", flush=True)
    with torch.no_grad():
        traced_out = traced(feat, cond, ray)
    max_diff = (ref_out - traced_out).abs().max().item()
    print(f"  Max diff (wrapper vs traced): {max_diff:.2e}", flush=True)
    if max_diff < 1e-4:
        print("  Traced model matches wrapper output. ✓", flush=True)
    else:
        print(f"  WARNING: Large difference {max_diff:.2e}", flush=True)

    # ── CoreML conversion ─────────────────────────────────────────────
    print("Converting to CoreML...", flush=True)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="features", shape=feat.shape),
            ct.TensorType(name="condition_info", shape=cond.shape),
            ct.TensorType(name="ray_cond", shape=ray.shape),
        ],
        outputs=[ct.TensorType(name="pose_token")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=(args.verify_only),
    )
    print(f"  Converted in {time.time()-t0:.1f}s", flush=True)

    if args.verify_only:
        print("Phase 1 PASSED: CoreML conversion is feasible.", flush=True)
        return

    # ── Save ──────────────────────────────────────────────────────────
    print(f"Saving to {args.out}...", flush=True)
    mlmodel.save(args.out)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(args.out) for f in fns
    ) / 1e6
    print(f"  Saved ({size_mb:.1f} MB)", flush=True)
    print("Done. ✓", flush=True)


if __name__ == "__main__":
    main()

