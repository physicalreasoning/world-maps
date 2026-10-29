"""Physion: measurement baselines and model readouts on the 1.5 s context that people saw.

Stages (``--stages``):
  detector   run the red-yellow contact detector on the REAL outcome frames of every stimulus (validation of the
             measurement against the ground-truth label) and the kinematic heuristic features of the context
             -> results/physion/detector_real.csv, results/physion/kinematics.csv            (CPU, about 5 min)
  encoders   frozen video/image encoders on the 1.5 s context (16 frames) -> results/physion/features/{model}.npz
  vlm        Qwen3-VL P(yes) to "will the red object touch the yellow area?" -> results/physion/vlm.csv

Models: vjepaL, vjepaH, vmaeL, dinoL, siglip2 (mean over 8 frames + last frame).
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from world_maps import physion as P  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402

OUT = RESULTS / "physion"
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MODELS = {
    "vjepaL": ("vjepa2", "facebook/vjepa2-vitl-fpc64-256", 256),
    "vjepaH": ("vjepa2", "facebook/vjepa2-vith-fpc64-256", 256),
    "vmaeL": ("videomae", "MCG-NJU/videomae-large", 224),
    "vmaeB": ("videomae", "MCG-NJU/videomae-base", 224),
    "dinoL": ("image", "facebook/dinov2-large", 224),
    "dinoB": ("image", "facebook/dinov2-base", 224),
    "siglip2": ("image", "google/siglip2-so400m-patch16-256", 256),
}
VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = ("This short clip shows a physical scene. One object is red and there is a yellow area or object. The clip stops before "
          "the outcome. Based on how things are moving, will the red object touch the yellow area or object after the clip ends? "
          "Answer Yes or No.")


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def context_frames(stim: P.Stim, size: int, k: int = 16) -> np.ndarray:
    fr = P.read_frames(stim.video, 0, P.CONTEXT_FRAMES, size=size)
    return fr[P.sample_indices(len(fr), k)]


def pool_grid(tok, side, grid=4):
    B, N, D = tok.shape
    x = tok.view(B, side, side, D).permute(0, 3, 1, 2)
    if side % grid and x.device.type == "mps":
        x = x.cpu()
    return F.adaptive_avg_pool2d(x, grid).permute(0, 2, 3, 1).reshape(B, -1)


# ----------------------------------------------------------------------------- detector + kinematics (CPU)

def stage_detector(stims, size=256, log=print):
    rows, kin = [], []
    t0 = time.time()
    for i, s in enumerate(stims):
        fr = P.read_frames(s.video, size=size)
        if len(fr) <= P.CONTEXT_FRAMES + 5:
            continue
        m = cv2.imread(str(s.map_png))
        cm = P.ColourModel(fr[0], cv2.cvtColor(m, cv2.COLOR_BGR2RGB) if m is not None else None)
        ctx, out = fr[: P.CONTEXT_FRAMES], fr[P.CONTEXT_FRAMES:]
        r = P.contact_score(out, cm, context=ctx)
        r_ctx = P.contact_score(ctx[-15:], cm, context=ctx[:30])
        rows.append({"stim_id": s.stim_id, "scenario": s.scenario, "label": s.label, "p_yes_human": s.p_yes_human, "n_human": s.n_human,
                     "n_frames": len(fr), **{f"real_{k}": v for k, v in r.items()}, "ctx_contact": r_ctx["contact"], "ctx_score": r_ctx["score"]})
        kin.append({"stim_id": s.stim_id, **{f"k{j}": v for j, v in enumerate(P.kinematic_features(ctx[::3], cm))}})
        if i % 100 == 0:
            log(f"  detector {i}/{len(stims)} ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(OUT / "detector_real.csv", index=False)
    pd.DataFrame(kin).to_csv(OUT / "kinematics.csv", index=False)
    df = pd.DataFrame(rows)
    acc = df.groupby("scenario").apply(lambda g: float((g.real_contact == g.label).mean()))
    log("detector accuracy on real outcomes by scenario:\n" + acc.round(2).to_string())


# ----------------------------------------------------------------------------- encoders

def stage_encoders(stims, names, log=print):
    (OUT / "features").mkdir(parents=True, exist_ok=True)
    meta_p = OUT / "features" / "models.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}
    for name in names:
        family, hub, size = MODELS[name]
        f = OUT / "features" / f"{name}.npz"
        if f.exists():
            log(f"{name}: done, skipping")
            continue
        t0 = time.time()
        try:
            feats = {}
            if family == "vjepa2":
                from transformers import VJEPA2Model

                try:
                    m = VJEPA2Model.from_pretrained(hub, attn_implementation="eager").to(DEV).eval()
                except (TypeError, ValueError):
                    m = VJEPA2Model.from_pretrained(hub).to(DEV).eval()
                dtype = torch.bfloat16 if DEV == "cuda" else torch.float32
                m = m.to(dtype)
                side = size // 16
                per = side * side
                rows_mean, rows_last, rows_grid = [], [], []
                for i, s in enumerate(stims):
                    x = torch.from_numpy(context_frames(s, size)).permute(0, 3, 1, 2).float() / 255.0
                    x = ((x - MEAN) / STD).unsqueeze(0).to(DEV, dtype)
                    with torch.no_grad():
                        h = m(x, skip_predictor=True).last_hidden_state.float()
                    rows_mean.append(h.mean(1).cpu().half().numpy()[0])
                    rows_last.append(h[:, -per:].mean(1).cpu().half().numpy()[0])
                    rows_grid.append(pool_grid(h[:, -per:], side).cpu().half().numpy()[0])
                    if i % 200 == 0:
                        log(f"  {name} {i}/{len(stims)}")
                feats = {"mean": np.stack(rows_mean), "last_slot": np.stack(rows_last), "last_grid": np.stack(rows_grid)}
            elif family == "videomae":
                from transformers import VideoMAEModel

                try:  # the fused attention kernel has raised illegal-memory-access errors for this model on some CUDA stacks
                    m = VideoMAEModel.from_pretrained(hub, attn_implementation="eager").to(DEV).eval()
                except TypeError:
                    m = VideoMAEModel.from_pretrained(hub).to(DEV).eval()
                side = size // 16
                per = side * side
                rows_mean, rows_grid = [], []
                for i, s in enumerate(stims):
                    x = torch.from_numpy(context_frames(s, size)).permute(0, 3, 1, 2).float() / 255.0
                    x = ((x - MEAN) / STD).unsqueeze(0).to(DEV)
                    with torch.no_grad():
                        h = m(pixel_values=x).last_hidden_state.float()
                    rows_mean.append(h.mean(1).cpu().half().numpy()[0])
                    rows_grid.append(pool_grid(h[:, -per:], side).cpu().half().numpy()[0])
                    if i % 200 == 0:
                        log(f"  {name} {i}/{len(stims)}")
                feats = {"mean": np.stack(rows_mean), "last_grid": np.stack(rows_grid)}
            else:
                from transformers import AutoModel

                try:  # fused attention kernels fault on some CUDA stacks; eager is slower but safe
                    m = AutoModel.from_pretrained(hub, attn_implementation="eager").to(DEV).eval()
                except (TypeError, ValueError):
                    m = AutoModel.from_pretrained(hub).to(DEV).eval()
                vision = m.vision_model if hasattr(m, "vision_model") else m
                mean, std = (torch.tensor([0.5] * 3).view(1, 3, 1, 1),) * 2 if "siglip" in hub else (MEAN, STD)
                rows_mean, rows_last, rows_diff = [], [], []
                for i, s in enumerate(stims):
                    fr = context_frames(s, size, k=8)
                    x = ((torch.from_numpy(fr).permute(0, 3, 1, 2).float() / 255.0 - mean) / std).to(DEV)
                    with torch.no_grad():
                        out = vision(pixel_values=x)
                        e = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
                        e = e.float()
                    rows_mean.append(e.mean(0).cpu().half().numpy())
                    rows_last.append(e[-1].cpu().half().numpy())
                    rows_diff.append((e[-1] - e[0]).cpu().half().numpy())
                    if i % 200 == 0:
                        log(f"  {name} {i}/{len(stims)}")
                feats = {"mean": np.stack(rows_mean), "last": np.stack(rows_last), "last_minus_first": np.stack(rows_diff)}
            np.savez(f, stim_id=np.array([s.stim_id for s in stims]), **{f"X_{k}": v for k, v in feats.items()})
            meta[name] = {"status": "ok", "hub": hub, "params": n_params(m), "seconds": round(time.time() - t0, 1), "n": len(stims),
                          "dims": {k: int(v.shape[1]) for k, v in feats.items()}}
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            meta[name] = {"status": f"failed: {type(e).__name__}: {str(e)[:200]}", "hub": hub}
            json.dump(meta, open(meta_p, "w"), indent=1)
            if "CUDA" in str(e) or "cuda" in type(e).__name__.lower():
                log(f"{name}: CUDA error; the CUDA context is unusable in this process. Rerun the stage: finished models are skipped.")
                break
        json.dump(meta, open(meta_p, "w"), indent=1)
        log(f"{name}: {meta[name]['status']}")


# ----------------------------------------------------------------------------- VLM

def stage_vlm(stims, hub=VLM_ID, log=print):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if DEV != "cpu" else torch.float32
    try:
        m = AutoModelForImageTextToText.from_pretrained(hub, dtype=dtype, device_map={"": DEV}).eval()
    except TypeError:
        m = AutoModelForImageTextToText.from_pretrained(hub, torch_dtype=dtype, device_map={"": DEV}).eval()
    proc = AutoProcessor.from_pretrained(hub)
    tok = proc.tokenizer
    yes_ids = {tok.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")}
    no_ids = {tok.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")}
    fps = 16 / 1.5
    rows = []
    out_f = OUT / "vlm.csv"
    done = set(pd.read_csv(out_f).stim_id) if out_f.exists() else set()
    for i, s in enumerate(stims):
        if s.stim_id in done:
            continue
        frames = context_frames(s, 256)
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = proc(text=[text], videos=[frames], return_tensors="pt",
                          video_metadata=[{"fps": fps, "total_num_frames": len(frames), "duration": len(frames) / fps}])
        except TypeError:
            inputs = proc(text=[text], videos=[frames], return_tensors="pt")
        inputs = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            logits = m(**inputs).logits[0, -1].float()
        ly, ln = torch.logsumexp(logits[list(yes_ids)], 0), torch.logsumexp(logits[list(no_ids)], 0)
        rows.append({"stim_id": s.stim_id, "p_yes": float(torch.softmax(torch.stack([ly, ln]), 0)[0])})
        if i % 100 == 0:
            log(f"  vlm {i}/{len(stims)}")
            pd.concat([pd.read_csv(out_f) if out_f.exists() else pd.DataFrame(), pd.DataFrame(rows)]).to_csv(out_f, index=False)
            rows = []
    pd.concat([pd.read_csv(out_f) if out_f.exists() else pd.DataFrame(), pd.DataFrame(rows)]).to_csv(out_f, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["detector", "encoders"])
    ap.add_argument("--models", nargs="+", default=["vjepaL", "vmaeL", "dinoL", "siglip2"])
    ap.add_argument("--vlm-id", default=VLM_ID)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute the detector stage even if its outputs exist")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    P.fetch_assets()
    stims = P.stimuli(with_human_only=True)
    if args.limit:
        stims = stims[: args.limit]
    P.stim_table(stims).to_csv(OUT / "stimuli.csv", index=False)
    if "detector" in args.stages:
        if (OUT / "detector_real.csv").exists() and (OUT / "kinematics.csv").exists() and not args.force and not args.limit:
            print("detector outputs exist (committed in the repository; same code and data); use --force to recompute")
        else:
            stage_detector(stims)
    if "encoders" in args.stages:
        stage_encoders(stims, args.models)
    if "vlm" in args.stages:
        stage_vlm(stims, args.vlm_id)


if __name__ == "__main__":
    main()
