"""Pilot 0 feature extraction: per-200ms-bin signals on the 32 physical-event base videos
(Yates, Yasuda & Yildirim, JEP:HPP 2024; OSF ma32p).

Bin k <-> slowdown-probe onset at t_k = 0.2*k s after video start (countdown excluded),
matching the lab's analysis notebook (iti = 200 ms; RT = (press - countdown) - k*200).

Signals (all computed only from the base video, without any slowdown):
  vjepa_surprise  V-JEPA 2 ViT-L latent prediction error: encoder sees ONLY the 12 frames
                  before t_k (15 fps, 0.8 s); predictor predicts the 4 frames after t_k;
                  target = encoder tokens of the real 16-frame clip (layer-normed), L1.
  vjepa_mismatch  same, but target taken from another scene at the same bin (sanity check).
  vjepa_surprise_top5  mean of the top 5% per-token errors (localized surprise; secondary).
  vjepa_change    1 - cos between mean encoder tokens of tubelets 4-5 (pre) and 6-7 (post).
  vmae_surprise   VideoMAE-base pixel reconstruction error of the 2 future tubelets with
                  the future fully masked (encoder sees only the past).
  dino_change     1 - cos of DINOv2-base CLS embeddings, frame at t_k-0.2 s vs t_k+0.2 s.
  pix_diff        mean |frame difference| over [t_k, t_k+0.2] at 30 fps (grayscale).
  flow_speed      mean Farneback flow magnitude over [t_k, t_k+0.2].
  flow_accel      |flow_speed(t_k..t_k+0.2) - flow_speed(t_k-0.2..t_k)|.
Output: features.csv (one row per scene x bin).
"""

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from world_maps.core import countdown_start, letterbox_square
from world_maps.paths import EVENT_VIDEOS, LENGTHS, RESULTS

ROOT = str(EVENT_VIDEOS)
LEN = json.load(open(LENGTHS))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
FPS_MODEL = 15
SRC_FPS = 120
torch.manual_seed(0)


def base_video(scene):
    """Original 120 fps render whose probe index lies beyond the clip end (no slowdown shown)."""
    files = glob.glob(f"{ROOT}/{scene}/non-visible/{scene}_125ms_*.mp4")
    if not files:  # one scene ships only countdown-prefixed 30 fps renders
        files = glob.glob(f"{ROOT}/{scene}/non-visible/final_{scene}_125ms_*.mp4")
    best = max(files, key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    k = int(best.rsplit("_", 1)[1].split(".")[0])
    assert k * 0.2 > LEN[scene], (scene, k, LEN[scene])
    return best


def load_frames(path, size):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps - SRC_FPS) < 1 or abs(fps - 30) < 1, fps
    rep = int(round(SRC_FPS / fps))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = letterbox_square(fr)  # keep the whole scene, no cropping
        out.append(cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA))
    out = np.stack(out)
    if os.path.basename(path).startswith("final_"):
        # strip the grey countdown: scene starts at the first large jump in frame contrast
        out = out[countdown_start(out) :]
    return np.repeat(out, rep, axis=0) if rep > 1 else out


def frame_at(frames, t):
    return frames[min(max(int(round(t * SRC_FPS)), 0), len(frames) - 1)]


def clip(frames, t_end, n, fps=FPS_MODEL):
    """n frames at `fps`, the last one just before t_end (strictly past-only when used as context)."""
    ts = t_end - (np.arange(n)[::-1] + 1) / fps
    return np.stack([frame_at(frames, t) for t in ts])


def clip_fwd(frames, t0, n, fps=FPS_MODEL):
    ts = t0 + np.arange(n) / fps
    return np.stack([frame_at(frames, t) for t in ts])


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(x):  # (T,H,W,3) uint8 -> (T,3,H,W) normalized
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


def main(scenes, out):
    from transformers import AutoModel, VideoMAEForPreTraining, VJEPA2Model

    vj = VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(DEV).eval()
    vm = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base").to(DEV).eval()
    dn = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
    per_tub = (256 // 16) ** 2  # 256 tokens per tubelet (V-JEPA 2, 256 px)
    vm_per_tub = (224 // 16) ** 2  # 196 (VideoMAE, 224 px)
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {r["scene"] for r in rows}
    for si, scene in enumerate(scenes):
        if scene in done:
            continue
        t0 = time.time()
        f256 = load_frames(base_video(scene), 256)
        f224 = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in f256])
        gray = np.stack([cv2.cvtColor(cv2.resize(f, (128, 128)), cv2.COLOR_RGB2GRAY) for f in f256])
        other = scenes[(si + 1) % len(scenes)]
        f_other = load_frames(base_video(other), 256)
        n_bins = int(np.floor(LEN[scene] / 0.2))
        # 30 fps flow / pixel series for low-level baselines
        idx30 = np.arange(0, len(gray), SRC_FPS // 30)
        g30 = gray[idx30].astype(np.float32)
        speed = [0.0]
        pdiff = [0.0]
        for a, b in zip(g30[:-1], g30[1:], strict=True):
            fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            speed.append(float(np.linalg.norm(fl, axis=2).mean()))
            pdiff.append(float(np.abs(b - a).mean()))
        speed, pdiff = np.array(speed), np.array(pdiff)

        def win(series, a, b):
            i0, i1 = int(round(a * 30)), int(round(b * 30))
            i0, i1 = max(i0, 1), max(min(i1, len(series)), 2)
            return float(series[i0:i1].mean()) if i1 > i0 else float("nan")

        for k in range(1, n_bins + 1):
            t = 0.2 * k
            row = dict(scene=scene, bin=k, t=t)
            row["pix_diff"] = win(pdiff, t, t + 0.2)
            row["flow_speed"] = win(speed, t, t + 0.2)
            row["flow_accel"] = abs(win(speed, t, t + 0.2) - win(speed, t - 0.2, t))
            with torch.no_grad():
                ctx = to_tensor(clip(f256, t, 12)).unsqueeze(0).to(DEV)
                full = torch.cat([ctx, to_tensor(clip_fwd(f256, t, 4)).unsqueeze(0).to(DEV)], 1)
                full_o = (
                    torch.cat([to_tensor(clip(f_other, t, 12)), to_tensor(clip_fwd(f_other, t, 4))], 0)
                    .unsqueeze(0)
                    .to(DEV)
                )
                enc_ctx = vj(ctx, skip_predictor=True).last_hidden_state  # (1, 6*256, 1024)
                enc_full = vj(full, skip_predictor=True).last_hidden_state  # (1, 8*256, 1024)
                enc_o = vj(full_o, skip_predictor=True).last_hidden_state
                cm = [torch.arange(6 * per_tub, device=DEV).unsqueeze(0)]
                tm = [torch.arange(6 * per_tub, 8 * per_tub, device=DEV).unsqueeze(0)]
                pred = vj.predictor(
                    encoder_hidden_states=enc_ctx, context_mask=cm, target_mask=tm
                ).last_hidden_state
                tgt = F.layer_norm(enc_full[:, 6 * per_tub :], (enc_full.size(-1),))
                tgt_o = F.layer_norm(enc_o[:, 6 * per_tub :], (enc_o.size(-1),))
                row["vjepa_surprise"] = float((pred - tgt).abs().mean())
                row["vjepa_mismatch"] = float((pred - tgt_o).abs().mean())
                tok = (pred - tgt).abs().mean(-1).flatten()  # per-token error
                tok_o = (pred - tgt_o).abs().mean(-1).flatten()
                q = max(1, int(0.05 * tok.numel()))
                row["vjepa_surprise_top5"] = float(tok.topk(q).values.mean())
                row["vjepa_mismatch_top5"] = float(tok_o.topk(q).values.mean())
                pre = enc_full[:, 4 * per_tub : 6 * per_tub].mean(1)
                post = enc_full[:, 6 * per_tub :].mean(1)
                row["vjepa_change"] = float(1 - F.cosine_similarity(pre, post).item())
                # VideoMAE: mask the two future tubelets entirely
                vclip = torch.cat([to_tensor(clip(f224, t, 12)), to_tensor(clip_fwd(f224, t, 4))], 0)
                vclip = vclip.unsqueeze(0).to(DEV)
                mask = torch.zeros(1, 8 * vm_per_tub, dtype=torch.bool, device=DEV)
                mask[:, 6 * vm_per_tub :] = True
                row["vmae_surprise"] = float(vm(pixel_values=vclip, bool_masked_pos=mask).loss)
                # DINOv2 frame change
                fa = to_tensor(
                    np.stack(
                        [
                            cv2.resize(frame_at(f256, t - 0.2), (224, 224)),
                            cv2.resize(frame_at(f256, t + 0.2), (224, 224)),
                        ]
                    )
                ).to(DEV)
                cls = dn(pixel_values=fa).last_hidden_state[:, 0]
                row["dino_change"] = float(1 - F.cosine_similarity(cls[:1], cls[1:]).item())
            rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{scene}: {n_bins} bins in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(RESULTS / "events_features.csv"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenes (smoke test)")
    args = ap.parse_args()
    main(sorted(LEN)[: args.limit], args.out)
