"""Build the Colab notebook for the shape-under-cloth study from the local source files.

Usage: python notebooks/build_cloth_notebook.py
Writes notebooks/colab_draped_shapes.ipynb (and the handoff copy if the path exists).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = {
    "src/world_maps/cloth.py": REPO / "src/world_maps/cloth.py",
    "scripts/cloth_extract.py": REPO / "scripts/cloth_extract.py",
    "scripts/cloth_analyze.py": REPO / "scripts/cloth_analyze.py",
    "scripts/cloth_imagine.py": REPO / "scripts/cloth_imagine.py",
}
HANDOFF = Path("/Users/lovejesh/Desktop/research/ilker-yildirim/colab/02_draped_shapes.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
    md("""# World Maps — shape under cloth: do modern models see the hidden object like people do? (Colab)

Yildirim et al. (2024, *Nat Hum Behav*): people match a cloth-draped object to its bare counterpart; a physics-based
analysis-by-synthesis model (PbAS) explains their trial-level accuracy and response times, while ImageNet CNNs sit at
chance on the hard trials. This notebook runs, on the same 120 trials and the same human data (public):

| Stage | What | Readout |
|---|---|---|
| Encoders | DINOv2 B/L/g, DINOv3-L, SigLIP 2, CLIP-L, V-JEPA 2 L/H (static clip), AlexNet fc6 (paper's reference) | paper protocol (correlation of embeddings), per cell accuracy, trial-level r with human accuracy (participant bootstrap), decision margin vs human RT |
| De-draping (new) | linear map draped → bare learned on other trials / other categories | is the hidden shape *linearly recoverable* even when raw similarity fails? are the errors human-like? |
| Imagination (new) | a learned image generator replaces the cloth simulator: drape the candidates / un-drape the target, compare in feature space; sequential sampling gives an RT prediction | generator-in-the-loop analysis-by-synthesis vs humans and vs PbAS |
| VLM | Qwen3-VL 2AFC | behavioural reference |

**Runtime:** GPU (A100 preferred). Encoders ≈ 10 min. VLM ≈ 10 min. Imagination with SDXL ≈ 25–40 min for K = 4
(960 + 480 images); FLUX Kontext ≈ 3–4 h. **When it finishes** the last cell downloads `world_maps_cloth_results.zip`.
Nothing is pushed to GitHub.
"""),
    code("""#@title 1. Configuration
MODELS = ["alexnet", "dinoB", "dinoL", "dinog", "dinov3L", "siglip2", "clipL", "vjepaL", "vjepaH"]  #@param {type:"raw"}
RUN_VLM = True            #@param {type:"boolean"}   # Qwen3-VL 2AFC (8B if the GPU has >= 20 GB)
RUN_IMAGINE = True        #@param {type:"boolean"}   # generator-in-the-loop stage
GENERATOR = "sdxl"        #@param ["sdxl", "kontext"]  # kontext is gated: accept the licence on the Hub + HF_TOKEN secret
K_SAMPLES = 4             #@param {type:"integer"}   # imagined samples per image
IMAGINE_SIZE = 768        #@param {type:"integer"}
IMAGINE_LIMIT = None      #@param {type:"raw"}       # e.g. 24 trials for a partial run; None = all 120
SMOKE_TEST_FIRST = True   #@param {type:"boolean"}
MOUNT_DRIVE = False       #@param {type:"boolean"}   # copy features and imagined images to Drive at the end

import os, torch
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (switch to a GPU runtime)'}  {gpu_gb:.0f} GB")
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    if tok: os.environ["HF_TOKEN"] = tok; print("HF_TOKEN loaded from Colab secrets")
except Exception:
    pass
if "dinov3L" in MODELS and not os.environ.get("HF_TOKEN"):
    print("note: dinov3L is gated on the Hub; without HF_TOKEN it will be recorded as failed and the rest continues")
print("models:", MODELS)
"""),
    code("""#@title 2. Code and dependencies (about 2 min)
%cd /content
import os
if not os.path.exists("world-maps"):
    !git clone -q https://github.com/physicalreasoning/world-maps.git
%cd /content/world-maps
!git rev-parse --short HEAD
!pip install -q "transformers>=5.0" accelerate "diffusers>=0.33" remotezip matplotlib pillow opencv-python-headless scikit-learn scipy pandas safetensors
import torch, transformers, torchvision, cv2, diffusers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| torchvision", torchvision.__version__, "| diffusers", diffusers.__version__)
os.environ["PYTHONPATH"] = "/content/world-maps/src"
os.makedirs("notebooks", exist_ok=True); os.makedirs("results/cloth", exist_ok=True)
"""),
]
for rel, path in FILES.items():
    cells.append(code(f"%%writefile /content/world-maps/{rel}\n" + path.read_text()))

cells += [
    code("""#@title 3. Public data: stimuli (51 MB) and behaviour (24 MB) from the lab's GitHub
%cd /content/world-maps
!PYTHONPATH=src python -c "from world_maps import cloth; cloth.fetch_assets(); import pandas as pd; df=cloth.load_behaviour(); print('trials', len(df), 'workers kept', df.workerid.nunique())"
!ls data/cloth/stimuli | head; ls data/cloth/stimuli/unoccluded-occluded | wc -l
"""),
    code("""#@title 4. Smoke test (DINOv2-base + analysis; 2 imagined images if enabled)
%cd /content/world-maps
if SMOKE_TEST_FIRST:
    !PYTHONPATH=src python scripts/cloth_extract.py --models dinoB 2>&1 | grep -v -i "warn\\|Loading weights\\|Fetching"
    !PYTHONPATH=src python scripts/cloth_analyze.py --n-boot 200 --out /content/smoke 2>&1 | tail -12
    if RUN_IMAGINE:
        !PYTHONPATH=src python scripts/cloth_imagine.py --generator {GENERATOR} --k 1 --limit 1 --size {IMAGINE_SIZE} --out /content/smoke_imagined 2>&1 | tail -3
        from IPython.display import display
        from PIL import Image
        import glob
        for f in sorted(glob.glob("/content/smoke_imagined/*/*.png")):
            print(f); display(Image.open(f).resize((160, 160)))
else:
    print("skipped")
"""),
    code("""#@title 5. Encoders on the 720 stimulus images (about 10 min)
%cd /content/world-maps
args = " ".join(MODELS)
!PYTHONPATH=src python scripts/cloth_extract.py --models {args} 2>&1 | grep -v -i "warn\\|Loading weights\\|Fetching"
import json; print(json.dumps({k: v.get("status") for k, v in json.load(open("results/cloth/features/models.json")).items()}, indent=1))
"""),
    code("""#@title 6. VLM 2AFC (optional, about 10 min)
%cd /content/world-maps
if RUN_VLM:
    !PYTHONPATH=src python scripts/cloth_extract.py --models vlm 2>&1 | grep -v -i "warn\\|Loading weights\\|Fetching" | tail -3
else:
    print("skipped")
"""),
    code("""#@title 7. Imagination: generate draped candidates and un-draped targets, then embed them (SDXL ≈ 30 min; Kontext ≈ 3–4 h)
%cd /content/world-maps
if RUN_IMAGINE:
    limit = f"--limit {IMAGINE_LIMIT}" if IMAGINE_LIMIT else ""
    !PYTHONPATH=src python scripts/cloth_imagine.py --generator {GENERATOR} --k {K_SAMPLES} --size {IMAGINE_SIZE} {limit} 2>&1 | tail -15
    # contact sheet: target | match | distractor | imagined draped match | imagined draped distractor | imagined bare object
    import glob, os
    import numpy as np
    from PIL import Image
    from IPython.display import display
    rows = []
    for t in [3, 9, 15, 27]:
        d = "data/cloth/stimuli/unoccluded-occluded"
        ims = [f"{d}/trial_{t}_gtCloth.png", f"{d}/trial_{t}_gtNoCloth.png", f"{d}/trial_{t}_disNoCloth0.png",
               f"results/cloth/imagined/drape/trial_{t}_match_0.png", f"results/cloth/imagined/drape/trial_{t}_distractor_0.png",
               f"results/cloth/imagined/undrape/trial_{t}_target_0.png"]
        ims = [np.array(Image.open(p).convert("RGB").resize((160, 160))) if os.path.exists(p) else np.zeros((160, 160, 3), np.uint8) for p in ims]
        rows.append(np.concatenate(ims, 1))
    sheet = Image.fromarray(np.concatenate(rows, 0)); sheet.save("results/cloth/imagined_contact_sheet.png"); display(sheet)
    # embed the imagined images with the same encoders (the extractor re-runs because the image table grew)
    args = " ".join(MODELS)
    !PYTHONPATH=src python scripts/cloth_extract.py --models {args} 2>&1 | grep -v -i "warn\\|Loading weights\\|Fetching" | tail -12
else:
    print("skipped")
"""),
    code("""#@title 8. Analysis (CPU, about 2 min): humans, PbAS reference, every model and layer, de-draping, imagination, VLM
%cd /content/world-maps
!PYTHONPATH=src python scripts/cloth_analyze.py --n-boot 2000
from IPython.display import Image as IM, display
for f in ["trial_correlation_occluded_same.png", "cell_accuracy.png", "layer_sweep.png"]:
    display(IM(filename=f"results/cloth/figures/{f}"))
"""),
    code("""#@title 9. Package results for download (features and imagined images go to Drive if enabled)
%cd /content/world-maps
!zip -q -r /content/world_maps_cloth_results.zip results/cloth/cloth_results.json results/cloth/cloth_results.md results/cloth/figures results/cloth/features/models.json results/cloth/features/image_index.csv results/cloth/imagined_contact_sheet.png scripts/cloth_extract.py scripts/cloth_analyze.py scripts/cloth_imagine.py src/world_maps/cloth.py 2>/dev/null
!ls -la /content/world_maps_cloth_results.zip
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    !mkdir -p /content/drive/MyDrive/world_maps_cloth && cp -r results/cloth/features results/cloth/imagined /content/drive/MyDrive/world_maps_cloth/ 2>/dev/null; ls /content/drive/MyDrive/world_maps_cloth
from google.colab import files
files.download("/content/world_maps_cloth_results.zip")
"""),
    md("""### If something fails
- **Gated models** (`dinov3L`, FLUX Kontext): accept the model terms on the Hub and add an `HF_TOKEN` secret; otherwise they are recorded as failed and the rest continues.
- **Out of memory on the VLM:** edit `MODELS["vlm"]` in `scripts/cloth_extract.py` to `Qwen/Qwen3-VL-4B-Instruct` and rerun cell 6.
- **Imagination too slow:** set `IMAGINE_LIMIT = 24` (first 24 trials: 2 same-category and 2 different-category blocks) or `K_SAMPLES = 2`; the analysis uses whatever exists (it needs all 120 trials for the imagination readouts, so partial runs only show images).
- **Runtime disconnects:** rerun cells 2, 3 and the stage that stopped; extraction and generation skip finished work.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                                  "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}, "nbformat": 4, "nbformat_minor": 5}
out = REPO / "notebooks" / "colab_draped_shapes.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
if HANDOFF.parent.exists():
    HANDOFF.write_text(json.dumps(nb, indent=1))
    print("wrote", HANDOFF)
