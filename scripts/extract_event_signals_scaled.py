"""Scaled and cross-family signals on the 32 physical-event base videos (Pilot 0 extension).

Same bins, clips and leak-free protocol as ``extract_event_signals.py`` (bin k <-> probe onset
t_k = 0.2 k s; context = the 12 frames before t_k at 15 fps; target = the 4 frames after t_k), run
on larger models and on other families. One model is loaded at a time; the merged table is
``results/events_features_scaled.csv`` (one row per scene x bin) and model metadata (ids,
parameter counts, device, dtype, seconds) go to ``results/events_models_scaled.json``.

Signals
  vjepa{H,g,g384}_surprise / _mismatch / _change   V-JEPA 2 ViT-H and ViT-g (256 px; g also at 384 px):
                                                   latent prediction error, other-scene control, pre/post change
  vmaeL_surprise                                    VideoMAE-large pixel reconstruction error, future tubelets masked
  dino{L,g}_change                                  DINOv2-large / -giant CLS change, frame t_k-0.2 vs t_k+0.2
  dinov3L_change                                    DINOv3 ViT-L/16 (gated on the Hub; needs HF_TOKEN), same readout
  siglip2_change                                    SigLIP 2 so400m image-embedding change, same frames
  vlm_boundary                                      Qwen3-VL: P("Yes" | "does a new physical event begin in the
                                                   last 0.3 s of this clip?") on the 16-frame clip around t_k
  vlm_unpredictable                                 Qwen3-VL: 1 - P("Yes" | "was the last 0.3 s predictable from
                                                   the earlier frames?")

The VLM readouts are a separate family: a language-grounded judgment of the clip, not a
predictive representation. They are reported next to the video-model signals, not pooled with them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_event_signals import FPS_MODEL, base_video, clip, clip_fwd, frame_at, load_frames, to_tensor  # noqa: E402

from world_maps.paths import LENGTHS, RESULTS  # noqa: E402

LEN = json.load(open(LENGTHS))
MEAN_IMNET = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_IMNET = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

# name -> (family, hub id, input size)
MODELS = {
    "vjepaH": ("vjepa2", "facebook/vjepa2-vith-fpc64-256", 256),
    "vjepag": ("vjepa2", "facebook/vjepa2-vitg-fpc64-256", 256),
    "vjepag384": ("vjepa2", "facebook/vjepa2-vitg-fpc64-384", 384),
    "vmaeL": ("videomae", "MCG-NJU/videomae-large", 224),
    "dinoL": ("dinov2", "facebook/dinov2-large", 224),
    "dinog": ("dinov2", "facebook/dinov2-giant", 224),
    "dinov3L": ("dinov3", "facebook/dinov3-vitl16-pretrain-lvd1689m", 224),
    "siglip2": ("siglip2", "google/siglip2-so400m-patch16-256", 256),
    "vlm": ("qwen_vl", "Qwen/Qwen3-VL-8B-Instruct", 256),
}
DEFAULT_SET = ["vjepaH", "vjepag", "vmaeL", "dinoL", "dinog", "siglip2", "vlm"]

PROMPT_BOUNDARY = (
    "This is a short clip of a simple physical scene with a few objects. In the final 0.3 seconds of the "
    "clip, does a new physical event begin, for example a collision, an object starting to fall, an object "
    "entering a container, or an object becoming hidden behind another? Answer Yes or No."
)
PROMPT_PREDICTABLE = (
    "This is a short clip of a simple physical scene with a few objects. Was what happened in the final "
    "0.3 seconds of the clip predictable from the earlier frames? Answer Yes or No."
)


def n_params(m) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def load_frames_cached(cache: dict, scene: str, size: int):
    if size not in cache:
        cache[size] = load_frames(base_video(scene), size)
    return cache[size]


# ---------------------------------------------------------------- per-family runners

def run_vjepa(name, hub, size, scenes, bins_of, log):
    from transformers import VJEPA2Model

    m = VJEPA2Model.from_pretrained(hub).to(DEV).eval()
    per_tub = (size // 16) ** 2
    rows = []
    for si, scene in enumerate(scenes):
        t0 = time.time()
        f = load_frames(base_video(scene), size)
        f_other = load_frames(base_video(scenes[(si + 1) % len(scenes)]), size)
        for k in bins_of(scene):
            t = 0.2 * k
            with torch.no_grad():
                ctx = to_tensor(clip(f, t, 12)).unsqueeze(0).to(DEV)
                full = torch.cat([ctx, to_tensor(clip_fwd(f, t, 4)).unsqueeze(0).to(DEV)], 1)
                full_o = torch.cat([to_tensor(clip(f_other, t, 12)), to_tensor(clip_fwd(f_other, t, 4))], 0).unsqueeze(0).to(DEV)
                enc_ctx = m(ctx, skip_predictor=True).last_hidden_state
                enc_full = m(full, skip_predictor=True).last_hidden_state
                enc_o = m(full_o, skip_predictor=True).last_hidden_state
                cm = [torch.arange(6 * per_tub, device=DEV).unsqueeze(0)]
                tm = [torch.arange(6 * per_tub, 8 * per_tub, device=DEV).unsqueeze(0)]
                pred = m.predictor(encoder_hidden_states=enc_ctx, context_mask=cm, target_mask=tm).last_hidden_state
                tgt = F.layer_norm(enc_full[:, 6 * per_tub:], (enc_full.size(-1),))
                tgt_o = F.layer_norm(enc_o[:, 6 * per_tub:], (enc_o.size(-1),))
                pre = enc_full[:, 4 * per_tub: 6 * per_tub].mean(1)
                post = enc_full[:, 6 * per_tub:].mean(1)
                rows.append({"scene": scene, "bin": k,
                             f"{name}_surprise": float((pred - tgt).abs().mean()),
                             f"{name}_mismatch": float((pred - tgt_o).abs().mean()),
                             f"{name}_change": float(1 - F.cosine_similarity(pre, post).item())})
        log(f"  {name} {scene}: {len(bins_of(scene))} bins in {time.time() - t0:.0f}s")
    return rows, n_params(m)


def run_videomae(name, hub, size, scenes, bins_of, log):
    from transformers import VideoMAEForPreTraining

    m = VideoMAEForPreTraining.from_pretrained(hub).to(DEV).eval()
    per_tub = (size // 16) ** 2
    rows = []
    for scene in scenes:
        t0 = time.time()
        f = load_frames(base_video(scene), size)
        for k in bins_of(scene):
            t = 0.2 * k
            with torch.no_grad():
                v = torch.cat([to_tensor(clip(f, t, 12)), to_tensor(clip_fwd(f, t, 4))], 0).unsqueeze(0).to(DEV)
                mask = torch.zeros(1, 8 * per_tub, dtype=torch.bool, device=DEV)
                mask[:, 6 * per_tub:] = True
                rows.append({"scene": scene, "bin": k, f"{name}_surprise": float(m(pixel_values=v, bool_masked_pos=mask).loss)})
        log(f"  {name} {scene}: {len(bins_of(scene))} bins in {time.time() - t0:.0f}s")
    return rows, n_params(m)


def run_image_change(name, family, hub, size, scenes, bins_of, log):
    """Frame-pair embedding change for image models (DINOv2, DINOv3, SigLIP 2), with explicit preprocessing
    (ImageNet mean/std for DINO; 0.5/0.5 for SigLIP) so no image-processor dependency is needed."""

    from transformers import AutoModel

    kw = {"token": os.environ.get("HF_TOKEN")} if os.environ.get("HF_TOKEN") else {}
    m = AutoModel.from_pretrained(hub, **kw).to(DEV).eval()
    if family == "siglip2":
        mean, std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    else:
        mean, std = MEAN_IMNET, STD_IMNET

    def embed(frames_uint8):
        x = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames_uint8])
        px = ((torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0 - mean) / std).to(DEV)
        with torch.no_grad():
            if family == "siglip2":
                vision = m.vision_model if hasattr(m, "vision_model") else m
                out = vision(pixel_values=px)
                return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state.mean(1)
            out = m(pixel_values=px)
            return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]

    rows = []
    for scene in scenes:
        t0 = time.time()
        f = load_frames(base_video(scene), max(size, 256))
        for k in bins_of(scene):
            t = 0.2 * k
            e = embed([frame_at(f, t - 0.2), frame_at(f, t + 0.2)])
            rows.append({"scene": scene, "bin": k, f"{name}_change": float(1 - F.cosine_similarity(e[:1], e[1:]).item())})
        log(f"  {name} {scene}: {len(bins_of(scene))} bins in {time.time() - t0:.0f}s")
    return rows, n_params(m)


def run_vlm(name, hub, size, scenes, bins_of, log):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if DEV != "cpu" else torch.float32
    try:
        m = AutoModelForImageTextToText.from_pretrained(hub, dtype=dtype, device_map={"": DEV}).eval()
    except TypeError:
        m = AutoModelForImageTextToText.from_pretrained(hub, torch_dtype=dtype, device_map={"": DEV}).eval()
    proc = AutoProcessor.from_pretrained(hub)
    tok = proc.tokenizer
    yes_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ("Yes", " Yes", "yes")}
    no_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ("No", " No", "no")}

    def p_yes(frames_uint8, prompt) -> float:
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = proc(text=[text], videos=[frames_uint8], return_tensors="pt",
                          video_metadata=[{"fps": FPS_MODEL, "total_num_frames": len(frames_uint8), "duration": len(frames_uint8) / FPS_MODEL}])
        except TypeError:
            inputs = proc(text=[text], videos=[frames_uint8], return_tensors="pt")
        inputs = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            logits = m(**inputs).logits[0, -1].float()
        ly = torch.logsumexp(logits[list(yes_ids)], 0)
        ln = torch.logsumexp(logits[list(no_ids)], 0)
        return float(torch.softmax(torch.stack([ly, ln]), 0)[0])

    rows = []
    for scene in scenes:
        t0 = time.time()
        f = load_frames(base_video(scene), size)
        for k in bins_of(scene):
            t = 0.2 * k
            frames = np.concatenate([clip(f, t, 12), clip_fwd(f, t, 4)], 0)  # 16 frames, 15 fps, uint8 (T,H,W,3)
            rows.append({"scene": scene, "bin": k,
                         f"{name}_boundary": p_yes(frames, PROMPT_BOUNDARY),
                         f"{name}_unpredictable": 1.0 - p_yes(frames, PROMPT_PREDICTABLE)})
        log(f"  {name} {scene}: {len(bins_of(scene))} bins in {time.time() - t0:.0f}s")
    return rows, n_params(m)


RUNNERS = {"vjepa2": run_vjepa, "videomae": run_videomae, "qwen_vl": run_vlm}


# ---------------------------------------------------------------- driver

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", default=DEFAULT_SET, help=f"subset of {list(MODELS)}")
    ap.add_argument("--vlm-id", default=None, help="override the Qwen3-VL hub id (e.g. Qwen/Qwen3-VL-4B-Instruct)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenes (smoke test)")
    ap.add_argument("--max-bins", type=int, default=None, help="only the first N bins per scene (smoke test)")
    ap.add_argument("--out", default=str(RESULTS / "events_features_scaled.csv"))
    ap.add_argument("--meta", default=str(RESULTS / "events_models_scaled.json"))
    args = ap.parse_args(argv)
    if args.vlm_id:
        MODELS["vlm"] = ("qwen_vl", args.vlm_id, 256)
    scenes = sorted(LEN)[: args.limit]

    def bins_of(scene):
        n = int(np.floor(LEN[scene] / 0.2))
        return list(range(1, (min(n, args.max_bins) if args.max_bins else n) + 1))

    merged = pd.read_csv(args.out) if os.path.exists(args.out) else pd.DataFrame(columns=["scene", "bin"])
    meta = json.load(open(args.meta)) if os.path.exists(args.meta) else {}
    log = lambda s: print(s, flush=True)  # noqa: E731
    log(f"device {DEV}; scenes {len(scenes)}; models {args.models}")
    for name in args.models:
        family, hub, size = MODELS[name]
        done_cols = [c for c in merged.columns if c.startswith(name + "_")]
        if done_cols and set(scenes) <= set(merged.dropna(subset=done_cols).scene.unique()):
            log(f"{name}: already in {args.out}, skipping")
            continue
        log(f"{name}: loading {hub}")
        t0 = time.time()
        try:
            if family in RUNNERS:
                rows, npar = RUNNERS[family](name, hub, size, scenes, bins_of, log)
            else:
                rows, npar = run_image_change(name, family, hub, size, scenes, bins_of, log)
        except Exception as e:  # noqa: BLE001  keep going with the other models; record the failure
            log(f"{name}: FAILED: {type(e).__name__}: {e}")
            meta[name] = {"hub": hub, "family": family, "status": f"failed: {type(e).__name__}: {str(e)[:300]}"}
            json.dump(meta, open(args.meta, "w"), indent=1)
            continue
        df = pd.DataFrame(rows)
        merged = merged.drop(columns=[c for c in merged.columns if c.startswith(name + "_")], errors="ignore")
        merged = df if merged.empty else merged.merge(df, on=["scene", "bin"], how="outer")
        merged.sort_values(["scene", "bin"]).to_csv(args.out, index=False)
        meta[name] = {"hub": hub, "family": family, "input_size": size, "params": npar, "device": DEV,
                      "seconds": round(time.time() - t0), "status": "ok", "scenes": len(scenes)}
        json.dump(meta, open(args.meta, "w"), indent=1)
        log(f"{name}: done in {time.time() - t0:.0f}s ({npar / 1e6:.0f}M params)")
        if DEV == "cuda":
            torch.cuda.empty_cache()
    log(f"wrote {args.out} and {args.meta}")


if __name__ == "__main__":
    main()
