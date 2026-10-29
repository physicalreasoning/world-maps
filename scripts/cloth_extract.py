"""Shape under cloth: embeddings of the 720 stimulus images (and of imagined images, if present).

For every model we store, per layer, one vector per image: the pooled token (CLS or mean) and the patch
tokens average-pooled to 4 x 4 (so a readout can use coarse spatial layout). Output
``results/cloth/features/{model}.npz`` with ``index`` columns (condition, trial, role, source, sample)
and arrays ``X_{layer}`` (float16). ``source`` is ``stim`` for the released images, ``drape`` / ``undrape``
for generator outputs written by ``cloth_imagine.py`` (``results/cloth/imagined/{drape,undrape,drape_video,undrape_video}/...``).

Models: alexnet (fc6, the paper's bottom-up reference; needs torchvision), dinoS/B/L/g (DINOv2),
dinov3L, siglip2, clipL, vjepaL/H/g (video encoder on a 16-frame static clip), and ``vlm`` (Qwen3-VL
2AFC: P(left) for "which object is under the cloth?", stored separately in ``vlm.npz``).
"""

from __future__ import annotations

import argparse
import glob
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
from world_maps import cloth  # noqa: E402
from world_maps.paths import RESULTS  # noqa: E402

OUT = RESULTS / "cloth" / "features"
IMAGINED = RESULTS / "cloth" / "imagined"
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)

MODELS = {
    "alexnet": ("alexnet", None, 224, {}),
    "dinoS": ("hf_image", "facebook/dinov2-small", 224, {"layers": [6, 9, 12]}),
    "dinoB": ("hf_image", "facebook/dinov2-base", 224, {"layers": [6, 9, 12]}),
    "dinoL": ("hf_image", "facebook/dinov2-large", 224, {"layers": [8, 12, 16, 20, 24]}),
    "dinog": ("hf_image", "facebook/dinov2-giant", 224, {"layers": [10, 20, 30, 40]}),
    "dinov3L": ("hf_image", "facebook/dinov3-vitl16-pretrain-lvd1689m", 224, {"layers": [8, 12, 16, 20, 24]}),
    "siglip2": ("hf_image", "google/siglip2-so400m-patch16-256", 256, {"layers": [9, 18, 27], "norm": "half"}),
    "clipL": ("hf_image", "openai/clip-vit-large-patch14", 224, {"layers": [8, 12, 16, 20, 24], "norm": "clip"}),
    "vjepaL": ("vjepa2", "facebook/vjepa2-vitl-fpc64-256", 256, {"layers": [8, 16, 24]}),
    "vjepaH": ("vjepa2", "facebook/vjepa2-vith-fpc64-256", 256, {"layers": [16, 24, 32]}),
    "vjepag": ("vjepa2", "facebook/vjepa2-vitg-fpc64-256", 256, {"layers": [20, 30, 40]}),
    "vlm": ("qwen_vl", "Qwen/Qwen3-VL-8B-Instruct", 224, {}),
}
DEFAULT = ["dinoB", "dinoL", "dinov3L", "siglip2", "clipL", "vjepaL", "alexnet"]


def n_params(m) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def load_rgb(path: str, size: int) -> np.ndarray:
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if im.shape[0] != size:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im


def pool_grid(tok: torch.Tensor, side: int, grid: int = 4) -> torch.Tensor:
    B, N, D = tok.shape
    x = tok.view(B, side, side, D).permute(0, 3, 1, 2)
    if side % grid and x.device.type == "mps":
        x = x.cpu()
    return F.adaptive_avg_pool2d(x, grid).permute(0, 2, 3, 1).reshape(B, -1)


def image_table() -> pd.DataFrame:
    idx = cloth.image_index()
    idx["source"], idx["sample"] = "stim", 0
    rows = [idx]
    for src in ["drape", "undrape", "drape_video", "undrape_video"]:
        for p in sorted(glob.glob(str(IMAGINED / src / "*.png"))):
            name = os.path.basename(p)[:-4]  # trial_{t}_{role}_{k}
            parts = name.split("_")
            rows.append(pd.DataFrame([{"condition": "occluded", "trial": int(parts[1]), "role": parts[2], "path": p,
                                       "same_category": ((int(parts[1]) // 6) % 2 == 0), "source": src, "sample": int(parts[3])}]))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- families

def run_hf_image(name, hub, size, opts, table, log):
    from transformers import AutoModel

    kw = {"token": os.environ.get("HF_TOKEN")} if os.environ.get("HF_TOKEN") else {}
    m = AutoModel.from_pretrained(hub, **kw).to(DEV).eval()
    vision = m.vision_model if hasattr(m, "vision_model") else m
    cfg = vision.config if hasattr(vision, "config") else m.config
    patch = getattr(cfg, "patch_size", 14)
    side = size // patch
    per = side * side
    if opts.get("norm") == "half":
        mean, std = torch.tensor([0.5] * 3).view(1, 3, 1, 1), torch.tensor([0.5] * 3).view(1, 3, 1, 1)
    elif opts.get("norm") == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = MEAN, STD
    layers = opts["layers"]
    feats = {f"L{L}_cls": [] for L in layers}
    feats.update({f"L{L}_grid": [] for L in layers})
    feats["final_pooled"] = []
    bs = 32
    paths = table.path.tolist()
    for s in range(0, len(paths), bs):
        x = np.stack([load_rgb(p, size) for p in paths[s: s + bs]])
        px = ((torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0 - mean) / std).to(DEV)
        with torch.no_grad():
            out = vision(pixel_values=px, output_hidden_states=True)
            hs = out.hidden_states
            for L in layers:
                h = hs[min(L, len(hs) - 1)].float()
                patches = h[:, -per:]
                cls = h[:, 0] if h.size(1) > per else patches.mean(1)
                feats[f"L{L}_cls"].append(cls.cpu().half().numpy())
                feats[f"L{L}_grid"].append(pool_grid(patches, side).cpu().half().numpy())
            pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
            feats["final_pooled"].append(pooled.float().cpu().half().numpy())
    log(f"  {name}: {len(paths)} images")
    return {k: np.concatenate(v) for k, v in feats.items()}, n_params(m)


def run_alexnet(name, hub, size, opts, table, log):
    import torchvision

    m = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1).to(DEV).eval()
    fc6 = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten(), *list(m.classifier.children())[:2])  # dropout, fc6
    fc6b = torch.nn.Sequential(fc6, *list(m.classifier.children())[2:5])  # relu, dropout, fc7
    feats = {"fc6": [], "fc7": [], "pool5": []}
    paths = table.path.tolist()
    for s in range(0, len(paths), 32):
        x = np.stack([load_rgb(p, size) for p in paths[s: s + 32]])
        px = ((torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD).to(DEV)
        with torch.no_grad():
            p5 = torch.nn.Flatten()(m.avgpool(m.features(px)))
            f6 = fc6(px)
            f7 = fc6b(px)
        feats["pool5"].append(p5.cpu().half().numpy())
        feats["fc6"].append(f6.cpu().half().numpy())
        feats["fc7"].append(f7.cpu().half().numpy())
    log(f"  {name}: {len(paths)} images")
    return {k: np.concatenate(v) for k, v in feats.items()}, n_params(m)


def run_vjepa(name, hub, size, opts, table, log):
    from transformers import VJEPA2Model

    m = VJEPA2Model.from_pretrained(hub).to(DEV).eval()
    dtype = torch.bfloat16 if DEV == "cuda" else torch.float32
    m = m.to(dtype)
    side = size // 16
    per = side * side
    layers = opts["layers"]
    feats = {f"L{L}_mean": [] for L in layers}
    feats.update({f"L{L}_grid": [] for L in layers})
    paths = table.path.tolist()
    for s in range(0, len(paths), 4):
        x = np.stack([load_rgb(p, size) for p in paths[s: s + 4]])
        px = ((torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD)
        clip = px.unsqueeze(1).repeat(1, 16, 1, 1, 1).to(DEV, dtype)  # static 16-frame clip
        with torch.no_grad():
            out = m(clip, skip_predictor=True, output_hidden_states=True)
            for L in layers:
                h = out.hidden_states[min(L, len(out.hidden_states) - 1)].float()[:, -per:]  # last temporal slot
                feats[f"L{L}_mean"].append(h.mean(1).cpu().half().numpy())
                feats[f"L{L}_grid"].append(pool_grid(h, side).cpu().half().numpy())
    log(f"  {name}: {len(paths)} images")
    return {k: np.concatenate(v) for k, v in feats.items()}, n_params(m)


def run_vlm(name, hub, size, opts, table, log):
    """Qwen3-VL 2AFC on the released trials: P('1') vs P('2') for which candidate is the object under the cloth
    (occluded) or the same object (unoccluded); candidates shown in both orders, averaged."""
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if DEV != "cpu" else torch.float32
    try:
        m = AutoModelForImageTextToText.from_pretrained(hub, dtype=dtype, device_map={"": DEV}).eval()
    except TypeError:
        m = AutoModelForImageTextToText.from_pretrained(hub, torch_dtype=dtype, device_map={"": DEV}).eval()
    proc = AutoProcessor.from_pretrained(hub)
    tok = proc.tokenizer
    ids1 = {tok.encode(t, add_special_tokens=False)[0] for t in ("1", " 1")}
    ids2 = {tok.encode(t, add_special_tokens=False)[0] for t in ("2", " 2")}
    prompts = {
        "occluded": "The first image shows an object completely covered by a thin draped cloth. The second and third images show "
                    "two different uncovered objects, each in its own random pose. Which uncovered object is the one under the cloth? "
                    "Answer with 1 for the second image or 2 for the third image.",
        "unoccluded": "The first image shows an object. The second and third images show two objects in other random poses. "
                      "Which one is the same object as in the first image? Answer with 1 for the second image or 2 for the third image.",
    }

    def p1(imgs, prompt):
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt")
        inputs = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            logits = m(**inputs).logits[0, -1].float()
        l1, l2 = torch.logsumexp(logits[list(ids1)], 0), torch.logsumexp(logits[list(ids2)], 0)
        return float(torch.softmax(torch.stack([l1, l2]), 0)[0])

    rows = []
    for tr in cloth.trials():
        for cond in cloth.CONDITIONS.values():
            im = tr.images(cond)
            T, M, D = (Image.open(im[k]).convert("RGB") for k in ("target", "match", "distractor"))
            p_match = 0.5 * (p1([T, M, D], prompts[cond]) + (1 - p1([T, D, M], prompts[cond])))
            rows.append({"condition": cond, "trial": tr.t, "p_match": p_match})
        log(f"  {name} trial {tr.t}")
    return pd.DataFrame(rows), n_params(m)


RUNNERS = {"hf_image": run_hf_image, "alexnet": run_alexnet, "vjepa2": run_vjepa, "qwen_vl": run_vlm}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out = __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cloth.fetch_assets()
    table = image_table()
    table.to_csv(out / "image_index.csv", index=False)
    meta_path = out / "models.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    def log(s):
        print(s, flush=True)

    for name in args.models:
        family, hub, size, opts = MODELS[name]
        f = out / f"{name}.npz"
        if f.exists() and not args.force and meta.get(name, {}).get("n_images") == len(table):
            log(f"{name}: done, skipping")
            continue
        t0 = time.time()
        try:
            res, npar = RUNNERS[family](name, hub, size, opts, table, log)
            if isinstance(res, pd.DataFrame):
                res.to_csv(out / f"{name}.csv", index=False)
                dims = {"p_match": 1}
            else:
                np.savez(f, **{f"X_{k}": v for k, v in res.items()})
                dims = {k: int(v.shape[1]) for k, v in res.items()}
            meta[name] = {"status": "ok", "hub": hub, "params": npar, "device": DEV, "n_images": len(table), "dims": dims,
                          "seconds": round(time.time() - t0, 1)}
            if DEV == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            meta[name] = {"status": f"failed: {type(e).__name__}: {str(e)[:300]}", "hub": hub}
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1)
        log(f"{name}: {meta[name]['status']} ({meta[name].get('seconds', 0)}s)")


if __name__ == "__main__":
    main()
