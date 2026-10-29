"""Imagination as simulation: a learned image generator in the loop of analysis-by-synthesis (GPU).

Two directions, both on the occluded condition's 120 trials:
  drape    for each candidate (match, distractor; bare object in its own pose) imagine K cloth-draped
           versions  ->  results/cloth/imagined/drape/trial_{t}_{match|distractor}_{k}.png
  undrape  for each draped target imagine K bare objects underneath
           ->  results/cloth/imagined/undrape/trial_{t}_target_{k}.png
``cloth_extract.py`` then embeds these images (source = drape / undrape) and ``cloth_analyze.py`` scores them
against the human data (imagination readouts, sequential-sampling RT).

Generators (diffusers):
  sdxl     stabilityai/stable-diffusion-xl-base-1.0 img2img (not gated; keeps silhouette and pose at moderate
           strength; about 2-4 s per image on an A100 at 768 px)
  kontext  black-forest-labs/FLUX.1-Kontext-dev instruction editing (gated: accept the licence on the Hub
           and set HF_TOKEN; about 10-20 s per image; better instruction following)
  ltx      LTX-Video 2B (Lightricks/LTX-Video-0.9.5; the 0.9.7/0.9.8 "dev" checkpoints are 13B, do not use them on a
           40 GB card) image-to-video: the cloth FALLS over the candidate (or is PULLED OFF the target) and the last
           frame is the imagined image; about 10-20 s per clip at 512 px. Directions are written as drape_video /
           undrape_video so the analysis can compare process (video) with appearance (image).
  wan      Wan-AI/Wan2.2-TI2V-5B-Diffusers image-to-video, same idea, slower (about 1-2 min per clip).
The prompts never name the object category, so no category information leaks into the imagined images.
Resumable: existing files are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from world_maps import cloth  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402

OUT = RESULTS / "cloth" / "imagined"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "drape": "the same object completely covered by a thin white cotton cloth draped over it, cloth falling naturally with folds and "
             "wrinkles, same camera angle, same pose, same size, matte grey material, uniform black background, 3D render",
    "undrape": "the rigid object that was hidden under the cloth, now uncovered, bare matte grey 3D render, same camera angle, same pose, "
               "same size, uniform black background",
}
NEGATIVE = "text, watermark, colour, people, hands, extra objects, blurry"
VIDEO = {
    "drape": "A thin white cotton cloth falls from above onto the grey object and drapes over it completely, settling with natural "
             "folds and wrinkles until the object is fully covered. Fixed camera, uniform black background, matte grey render, no "
             "other objects.",
    "undrape": "The white cloth covering the object is lifted away and slides off, revealing the bare rigid grey object underneath "
               "in the same position. Fixed camera, uniform black background, matte grey render, no other objects.",
}
VIDEO_DIRS = {"drape": "drape_video", "undrape": "undrape_video"}
KONTEXT = {
    "drape": "Cover this object completely with a thin white cotton cloth draped over it so that it is fully hidden; keep the camera "
             "angle, pose, size and the black background exactly the same.",
    "undrape": "Remove the cloth and show the rigid grey object that was underneath it; keep the camera angle, pose, size and the "
               "black background exactly the same.",
}


def load_pipe(kind: str):
    if kind == "sdxl":
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                                variant="fp16", use_safetensors=True).to(DEV)
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        return pipe
    if kind == "kontext":
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16,
                                                   token=os.environ.get("HF_TOKEN"))
        pipe.enable_model_cpu_offload()
        return pipe
    if kind == "ltx":
        from diffusers import LTXImageToVideoPipeline

        pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16).to(DEV)
        for fn in ("enable_tiling", "enable_slicing"):
            if hasattr(pipe.vae, fn):
                getattr(pipe.vae, fn)()
        print(f"LTX transformer: {sum(p.numel() for p in pipe.transformer.parameters()) / 1e9:.1f}B parameters", flush=True)
        return pipe
    if kind == "wan":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        rep = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(rep, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(rep, vae=vae, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe
    raise ValueError(kind)


def generate(pipe, kind: str, direction: str, image: Image.Image, seed: int, size: int, strength: float, steps: int,
             n_frames: int = 49, keep_video: str | None = None) -> Image.Image:
    g = torch.Generator(device="cpu").manual_seed(seed)
    src = image.convert("RGB").resize((size, size), Image.BICUBIC)
    if kind == "sdxl":
        return pipe(prompt=PROMPTS[direction], negative_prompt=NEGATIVE, image=src, strength=strength, guidance_scale=6.0,
                    num_inference_steps=steps, generator=g).images[0]
    if kind == "kontext":
        return pipe(image=src, prompt=KONTEXT[direction], guidance_scale=2.5, num_inference_steps=steps, height=size, width=size,
                    generator=g).images[0]
    # image-to-video: the last frame is the imagined end state of the physical process
    kw = dict(image=src, prompt=VIDEO[direction], negative_prompt=NEGATIVE + ", camera motion, zoom", width=size, height=size,
              num_frames=n_frames, num_inference_steps=steps, generator=g)
    if kind == "wan":
        kw["guidance_scale"] = 5.0
    frames = pipe(**kw).frames[0]
    if keep_video:
        try:
            from diffusers.utils import export_to_video

            export_to_video([f if isinstance(f, Image.Image) else Image.fromarray(f) for f in frames], keep_video, fps=24)
        except Exception as e:  # inspection only
            print(f"  (video export skipped: {type(e).__name__})", flush=True)
    last = frames[-1]
    return last if isinstance(last, Image.Image) else Image.fromarray(last)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="sdxl", choices=["sdxl", "kontext", "ltx", "wan"])
    ap.add_argument("--n-frames", type=int, default=49, help="frames per clip for the video generators (8k+1 for LTX, 4k+1 for Wan)")
    ap.add_argument("--keep-videos", type=int, default=6, help="save the first N clips per direction as mp4 for inspection")
    ap.add_argument("--directions", nargs="+", default=["drape", "undrape"])
    ap.add_argument("--k", type=int, default=4, help="imagined samples per image")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=None, help="denoising steps (default 30 for image models, 25 for video)")
    ap.add_argument("--strength-drape", type=float, default=0.6)
    ap.add_argument("--strength-undrape", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="first N trials (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = __import__("pathlib").Path(args.out)
    if args.steps is None:
        args.steps = 25 if args.generator in ("ltx", "wan") else 30
    cloth.fetch_assets()
    pipe = load_pipe(args.generator)
    trials = cloth.trials()[: args.limit] if args.limit else cloth.trials()
    t0 = time.time()
    n = 0
    is_video = args.generator in ("ltx", "wan")
    if is_video and args.size % 32:
        args.size = 32 * (args.size // 32)
        print(f"size rounded to {args.size} for the video generator")
    for direction in args.directions:
        d = out / (VIDEO_DIRS[direction] if is_video else direction)
        d.mkdir(parents=True, exist_ok=True)
        roles = ["match", "distractor"] if direction == "drape" else ["target"]
        strength = args.strength_drape if direction == "drape" else args.strength_undrape
        for tr in trials:
            ims = tr.images("occluded")
            for role in roles:
                for k in range(args.k):
                    f = d / f"trial_{tr.t}_{role}_{k}.png"
                    if f.exists():
                        continue
                    try:
                        keep = str(f.with_suffix(".mp4")) if (is_video and n < args.keep_videos) else None
                        img = generate(pipe, args.generator, direction, Image.open(ims[role]), seed=1000 * tr.t + 10 * k + (role == "distractor"),
                                       size=args.size, strength=strength, steps=args.steps, n_frames=args.n_frames, keep_video=keep)
                        img.resize((224, 224), Image.LANCZOS).save(f)
                        n += 1
                    except Exception as e:  # keep going
                        print(f"  failed {f.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            if tr.t % 10 == 0:
                print(f"{direction} trial {tr.t}: {n} images so far, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new images in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
