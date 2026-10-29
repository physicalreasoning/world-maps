"""Physion: generators as simulators. Continue the 1.5 s context with a video generator and read the imagined outcome.

For a stratified subset of stimuli (``--per-scenario`` per scenario, balanced on the label), the generator receives the
context frames people saw and produces the following seconds; the red-yellow contact detector (calibrated on the real
first frame and the real context) is run on the generated frames. Output: results/physion/generated/{generator}/scores.csv
with one row per (stimulus, sample), the detector's decision and soft score, plus a few mp4s and last frames for
inspection.

Generators (diffusers):
  ltx     Lightricks/LTX-Video-0.9.7-dev with video conditioning (LTXConditionPipeline): the context frames are the
          condition at frame 0; 97 frames at 24 fps are generated (about 2.7 s beyond the context). ~20-40 s per clip.
  cosmos  nvidia/Cosmos-Predict2-2B-Video2World (video-conditioned world model; experimental in this script).
The prompt describes a camera-fixed physical scene and never states the outcome.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from world_maps import physion as P  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402

OUT = RESULTS / "physion" / "generated"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = ("A fixed camera films a simple indoor physics scene with a few rigid objects on the floor. The objects keep moving "
          "according to gravity, momentum and collisions, exactly as they were moving. Realistic rendering, no camera motion, "
          "no new objects, no cuts.")
NEGATIVE = "camera motion, zoom, cut, text, watermark, new objects, people, hands, blur, flicker"


def select(stims, per_scenario: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for scen in P.SCENARIOS:
        sub = [s for s in stims if s.scenario == scen]
        for lab in (True, False):
            pool = [s for s in sub if s.label == lab]
            idx = rng.permutation(len(pool))[: per_scenario // 2]
            out += [pool[i] for i in sorted(idx)]
    return out


def load_pipe(kind: str):
    if kind == "ltx":
        from diffusers import LTXConditionPipeline

        pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16).to(DEV)
        return pipe
    if kind == "cosmos":
        from diffusers import Cosmos2VideoToWorldPipeline

        return Cosmos2VideoToWorldPipeline.from_pretrained("nvidia/Cosmos-Predict2-2B-Video2World", torch_dtype=torch.bfloat16).to(DEV)
    raise ValueError(kind)


def continue_video(pipe, kind: str, context_rgb: np.ndarray, seed: int, size: int, n_frames: int, steps: int) -> np.ndarray:
    """Return generated frames (T, size, size, 3) uint8 INCLUDING the conditioning part at the start (LTX)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    frames = [Image.fromarray(f).resize((size, size), Image.BICUBIC) for f in context_rgb]
    if kind == "ltx":
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        n_cond = 8 * ((len(frames) - 1) // 8) + 1  # LTX wants 8k+1 conditioning frames
        cond = LTXVideoCondition(video=frames[-n_cond:], frame_index=0)
        out = pipe(conditions=[cond], prompt=PROMPT, negative_prompt=NEGATIVE, width=size, height=size, num_frames=n_frames,
                   num_inference_steps=steps, generator=g).frames[0]
    else:
        out = pipe(video=frames[-9:], prompt=PROMPT, negative_prompt=NEGATIVE, height=size, width=size, num_frames=n_frames,
                   num_inference_steps=steps, generator=g).frames[0]
    arr = [np.array(f.convert("RGB")) if isinstance(f, Image.Image) else (f * 255).astype(np.uint8) if f.dtype != np.uint8 else f for f in out]
    return np.stack(arr)


def write_mp4(frames, path, fps=24):
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="ltx", choices=["ltx", "cosmos"])
    ap.add_argument("--per-scenario", type=int, default=40)
    ap.add_argument("--k", type=int, default=1, help="samples per stimulus")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n-frames", type=int, default=97)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--context-fps", type=float, default=24.0, help="context frames are resampled to the generator's frame rate")
    ap.add_argument("--keep-videos", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N selected stimuli (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    P.fetch_assets()
    stims = select(P.stimuli(with_human_only=True), args.per_scenario)
    if args.limit:
        stims = stims[: args.limit]
    out = __import__("pathlib").Path(args.out) / args.generator
    (out / "frames").mkdir(parents=True, exist_ok=True)
    scores_f = out / "scores.csv"
    done = set()
    if scores_f.exists():
        d = pd.read_csv(scores_f)
        done = set(zip(d.stim_id, d["sample"]))
    pipe = load_pipe(args.generator)
    t0 = time.time()
    n_kept = 0
    rows = []
    for i, s in enumerate(stims):
        full = P.read_frames(s.video, size=args.size)
        ctx = full[: P.CONTEXT_FRAMES]
        m = cv2.imread(str(s.map_png))
        cm = P.ColourModel(full[0], cv2.cvtColor(m, cv2.COLOR_BGR2RGB) if m is not None else None)
        # resample the 30 fps context to the generator's frame rate
        n_ctx = int(round(P.CONTEXT_FRAMES / P.FPS * args.context_fps))
        ctx_rs = ctx[P.sample_indices(len(ctx), n_ctx)]
        for k in range(args.k):
            if (s.stim_id, k) in done:
                continue
            try:
                gen = continue_video(pipe, args.generator, ctx_rs, seed=10 * i + k, size=args.size, n_frames=args.n_frames, steps=args.steps)
                n_cond = 8 * ((len(ctx_rs) - 1) // 8) + 1 if args.generator == "ltx" else 9
                future = gen[n_cond:] if len(gen) > n_cond + 4 else gen
                r = P.contact_score(future, cm, context=ctx)
                rows.append({"stim_id": s.stim_id, "scenario": s.scenario, "label": s.label, "p_yes_human": s.p_yes_human, "sample": k,
                             "n_generated": int(len(future)), **{f"gen_{kk}": v for kk, v in r.items()}})
                Image.fromarray(future[-1]).resize((256, 256)).save(out / "frames" / f"{s.stim_id}_{k}_last.png")
                if n_kept < args.keep_videos:
                    write_mp4(gen, out / f"{s.stim_id}_{k}.mp4")
                    n_kept += 1
            except Exception as e:
                print(f"  failed {s.stim_id} sample {k}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        if rows and (i % 5 == 0 or i == len(stims) - 1):
            prev = pd.read_csv(scores_f) if scores_f.exists() else pd.DataFrame()
            pd.concat([prev, pd.DataFrame(rows)]).to_csv(scores_f, index=False)
            rows = []
            print(f"{args.generator}: {i + 1}/{len(stims)} stimuli, {time.time() - t0:.0f}s", flush=True)
    if rows:
        prev = pd.read_csv(scores_f) if scores_f.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(rows)]).to_csv(scores_f, index=False)
    print(f"done in {time.time() - t0:.0f}s -> {scores_f}")


if __name__ == "__main__":
    main()
