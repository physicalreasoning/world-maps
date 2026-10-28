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
        return pipe
    if kind == "kontext":
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16,
                                                   token=os.environ.get("HF_TOKEN"))
        pipe.enable_model_cpu_offload()
        return pipe
    raise ValueError(kind)


def generate(pipe, kind: str, direction: str, image: Image.Image, seed: int, size: int, strength: float, steps: int) -> Image.Image:
    g = torch.Generator(device="cpu").manual_seed(seed)
    src = image.convert("RGB").resize((size, size), Image.BICUBIC)
    if kind == "sdxl":
        return pipe(prompt=PROMPTS[direction], negative_prompt=NEGATIVE, image=src, strength=strength, guidance_scale=6.0,
                    num_inference_steps=steps, generator=g).images[0]
    return pipe(image=src, prompt=KONTEXT[direction], guidance_scale=2.5, num_inference_steps=steps, height=size, width=size,
                generator=g).images[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="sdxl", choices=["sdxl", "kontext"])
    ap.add_argument("--directions", nargs="+", default=["drape", "undrape"])
    ap.add_argument("--k", type=int, default=4, help="imagined samples per image")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength-drape", type=float, default=0.6)
    ap.add_argument("--strength-undrape", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="first N trials (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = __import__("pathlib").Path(args.out)
    cloth.fetch_assets()
    pipe = load_pipe(args.generator)
    trials = cloth.trials()[: args.limit] if args.limit else cloth.trials()
    t0 = time.time()
    n = 0
    for direction in args.directions:
        d = out / direction
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
                        img = generate(pipe, args.generator, direction, Image.open(ims[role]), seed=1000 * tr.t + 10 * k + (role == "distractor"),
                                       size=args.size, strength=strength, steps=args.steps)
                        img.resize((224, 224), Image.LANCZOS).save(f)
                        n += 1
                    except Exception as e:  # keep going
                        print(f"  failed {f.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            if tr.t % 10 == 0:
                print(f"{direction} trial {tr.t}: {n} images so far, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new images in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
