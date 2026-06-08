"""Export SAM-3D-Body decoder (CameraEncoder + PromptEncoder + PromptableDecoder) to CoreML.

prompt_encoder.forward is bypassed entirely: at C++ inference time all keypoints are
label=-2 (no keypoints), so _embed_keypoints always returns zeros. Bypassing avoids
the data-dependent assert (points.min() >= 0) which EXIR cannot trace.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
FAST_REPO = os.path.join(ROOT, "..", "Fast-SAM-3D-Body")
sys.path.insert(0, FAST_REPO)

os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")

BACKBONE_DIM = 1280
DECODER_DIM = 1024


class CoreMLDecoderWrapper(nn.Module):
    """Wraps CameraEncoder + PromptEncoder + PromptableDecoder for CoreML export.

    Inputs:  features [B,1280,32,32], cond_info [B,3], ray_cond [B,2,32,32]
    Output:  pose_token [B,1024]
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

        for attr in (
            "keypoint_embedding",
            "keypoint3d_embedding",
            "hand_box_embedding",
        ):
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

        self.decoder.do_interm_preds = False
        self.decoder.keypoint_token_update = None
        
        self.register_buffer("img_pe", self.prompt_encoder.get_dense_pe((self.feat_hw, self.feat_hw)))

    def forward(self, features, cond_info, ray_cond):
        B = features.shape[0]
        H = W = self.feat_hw

        # CameraEncoder
        rays = ray_cond.permute(0, 2, 3, 1)
        rays = torch.cat([rays, torch.ones_like(rays[..., :1])], -1)
        rays_emb = self.ray_cond_emb.camera(pos=rays.reshape(B, H * W, 3))
        rays_emb = rays_emb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        z = torch.cat([features, rays_emb], dim=1)
        features = self.ray_cond_emb.norm(self.ray_cond_emb.conv(z))

        # Initial estimate
        init_pose = self.init_pose.weight.unsqueeze(0).expand(B, -1, -1)
        init_camera = self.init_camera.weight.unsqueeze(0).expand(B, -1, -1)
        init_est = torch.cat([init_pose, init_camera], dim=-1)

        # Pose token
        init_input = torch.cat([cond_info.unsqueeze(1), init_est], dim=-1)
        token_seq = self.init_to_token(init_input)

        # Previous-estimate token
        prev_emb = self.prev_to_token(init_est)

        # At C++ inference time keypoints are always label=-2 (no keypoints).
        # _embed_keypoints returns zeros for label=-2, but its assert (points.min() >= 0)
        # is a data-dependent branch that EXIR cannot trace. Bypass prompt_encoder.forward
        # entirely and pass the known-zero result directly to prompt_to_token.
        zero_kp = torch.zeros(
            B, 1, BACKBONE_DIM, dtype=features.dtype, device=features.device
        )
        prompt_emb = self.prompt_to_token(zero_kp)

        token_seq = torch.cat([token_seq, prev_emb, prompt_emb], dim=1)
        tok_aug = torch.zeros_like(token_seq)
        tok_aug[:, 1] = prev_emb[:, 0]
        tok_aug[:, 2] = prompt_emb[:, 0]

        for attr in (
            "keypoint_embedding",
            "keypoint3d_embedding",
            "hand_box_embedding",
        ):
            if hasattr(self, attr):
                emb = getattr(self, attr).weight.unsqueeze(0).expand(B, -1, -1)
                token_seq = torch.cat([token_seq, emb], dim=1)
                tok_aug = torch.cat([tok_aug, torch.zeros_like(emb)], dim=1)

        img_pe = self.img_pe.expand(B, -1, -1, -1)

        out = self.decoder(
            token_embedding=token_seq,
            image_embedding=features,
            token_augment=tok_aug,
            image_augment=img_pe,
            token_mask=None,
            channel_first=True,
            token_to_pose_output_fn=None,
        )
        token_out = out[0] if isinstance(out, (tuple, list)) else out
        return token_out[:, 0]


def load_decoder(checkpoint_dir: str, size: int) -> CoreMLDecoderWrapper:
    from sam_3d_body.build_models import load_sam_3d_body

    ckpt = os.path.join(checkpoint_dir, "model.ckpt")
    mhr = os.path.join(checkpoint_dir, "assets", "mhr_model.pt")
    model, _ = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr, device="cpu")
    return CoreMLDecoderWrapper(model, size).eval().float()


def smoke_forward(wrapper: CoreMLDecoderWrapper, size: int):
    feat_hw = size // 16
    feat = torch.randn(1, BACKBONE_DIM, feat_hw, feat_hw)
    cond = torch.randn(1, 3)
    ray = torch.randn(1, 2, feat_hw, feat_hw)
    t0 = time.time()
    with torch.no_grad():
        out = wrapper(feat, cond, ray)
    print(
        "forward %.2fs shape=%s dtype=%s mean=%.6f"
        % (time.time() - t0, tuple(out.shape), out.dtype, out.float().mean().item()),
        flush=True,
    )
    return feat, cond, ray


def convert_coreml(
    wrapper: CoreMLDecoderWrapper,
    example_inputs: tuple,
    out_path: str,
    size: int,
    verify_only: bool = False,
):
    import coremltools as ct

    feat, cond, ray = example_inputs
    feat_hw = size // 16

    print("tracing B=1...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        exported = torch.jit.trace(wrapper, (feat, cond, ray))
    print("traced %.2fs" % (time.time() - t0), flush=True)

    print("converting to CoreML MLProgram fp16...", flush=True)
    t0 = time.time()
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(
                name="features",
                shape=ct.Shape(
                    shape=(1, BACKBONE_DIM, feat_hw, feat_hw)
                ),
            ),
            ct.TensorType(
                name="condition_info", shape=ct.Shape(shape=(1, 3))
            ),
            ct.TensorType(
                name="ray_cond",
                shape=ct.Shape(shape=(1, 2, feat_hw, feat_hw)),
            ),
        ],
        outputs=[ct.TensorType(name="pose_token")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
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
        description="Export SAM-3D-Body decoder to CoreML via EXIR"
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=os.path.join(ROOT, "checkpoints", "sam-3d-body-dinov3"),
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "decoder_coreml.mlpackage"))
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify conversion feasibility, don't save",
    )
    ap.add_argument(
        "--size", type=int, default=512, help="Original backbone input resolution"
    )
    args = ap.parse_args()

    wrapper = load_decoder(args.checkpoint_dir, args.size)
    example_inputs = smoke_forward(wrapper, args.size)
    convert_coreml(
        wrapper, example_inputs, args.out, args.size, verify_only=args.verify_only
    )


if __name__ == "__main__":
    main()
