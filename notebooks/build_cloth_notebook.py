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

META = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}
nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}
out = REPO / "notebooks" / "colab_draped_shapes.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
if HANDOFF.parent.exists():
    HANDOFF.write_text(json.dumps(nb, indent=1))
    print("wrote", HANDOFF)

# ------------------------------------------------------------------ notebook 03: imagination as a physical process (video)
cells3 = [
    md("""# World Maps — imagination as a physical *process*: video generators inside analysis-by-synthesis (Colab)

Notebook 02 lets an image generator *edit* a candidate into a draped version. This notebook asks whether *simulating the
process* does better: an image-to-video model watches a cloth **fall onto** each candidate (or get **pulled off** the
target), and the last frame is the imagined end state. Same trials, same human data, same scoring as notebook 02, so
the two routes (appearance edit vs physical process) are compared directly against humans and PbAS.

| Stage | What |
|---|---|
| Video imagination | LTX-Video 2B (default, ≈ 10–20 s per clip at 512 px; the 13B "dev" checkpoints are not used) or Wan 2.2 TI2V-5B (≈ 1–2 min, optional) image-to-video; K clips per image; last frame saved; a few mp4s kept for inspection |
| Encoders | the models in `MODELS` embed the stimuli and the imagined frames |
| Analysis | humans, PbAS reference, raw readouts, de-draping, image-imagination (if present from notebook 02 on Drive) and video-imagination readouts with the sequential-sampling RT model |

**Runtime:** A100. K = 2 with LTX 2B at 512 px ≈ 1.5–2.5 h for 480 + 240 clips (expect about 12 GB of GPU memory);
set `IMAGINE_LIMIT` for a partial run (the analysis needs all 120 trials to score, partial runs are for inspecting quality). The last cell downloads
`world_maps_cloth_video_results.zip`. Nothing is pushed to GitHub.
"""),
    code("""#@title 1. Configuration
MODELS = ["dinoB", "dinoL", "siglip2", "clipL"]  #@param {type:"raw"}
VIDEO_GENERATOR = "ltx"     #@param ["ltx", "wan"]
K_SAMPLES = 2               #@param {type:"integer"}
VIDEO_SIZE = 512            #@param {type:"integer"}   # multiple of 32
N_FRAMES = 49               #@param {type:"integer"}   # 8k+1 for LTX, 4k+1 for Wan
DIRECTIONS = ["drape", "undrape"]  #@param {type:"raw"}
IMAGINE_LIMIT = None        #@param {type:"raw"}       # e.g. 12 trials to inspect quality first
SMOKE_TEST_FIRST = True     #@param {type:"boolean"}
LOAD_FROM_DRIVE = False     #@param {type:"boolean"}   # reuse results/cloth/features and imagined images saved by notebook 02

import os, torch
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (switch to a GPU runtime)'}  {gpu_gb:.0f} GB")
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    if tok: os.environ["HF_TOKEN"] = tok; print("HF_TOKEN loaded from Colab secrets")
except Exception:
    pass
"""),
    code("""#@title 2. Code and dependencies (about 2 min)
%cd /content
import os
if not os.path.exists("world-maps"):
    !git clone -q https://github.com/physicalreasoning/world-maps.git
%cd /content/world-maps
!git pull -q; git rev-parse --short HEAD
!pip install -q "transformers>=5.0" accelerate "diffusers>=0.33" imageio imageio-ffmpeg remotezip matplotlib pillow opencv-python-headless scikit-learn scipy pandas safetensors
import torch, transformers, diffusers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| diffusers", diffusers.__version__)
os.environ["PYTHONPATH"] = "/content/world-maps/src"
os.makedirs("results/cloth", exist_ok=True)
if LOAD_FROM_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    !cp -r /content/drive/MyDrive/world_maps_cloth/features results/cloth/ 2>/dev/null; cp -r /content/drive/MyDrive/world_maps_cloth/imagined results/cloth/ 2>/dev/null; ls results/cloth
!PYTHONPATH=src python -c "from world_maps import cloth; cloth.fetch_assets(); print('assets ready')"
"""),
]
for rel, path in FILES.items():
    cells3.append(code(f"%%writefile /content/world-maps/{rel}\n" + path.read_text()))
cells3 += [
    code("""#@title 3. Smoke test: one clip per direction, shown as video + last frame
%cd /content/world-maps
if SMOKE_TEST_FIRST:
    !PYTHONPATH=src python scripts/cloth_imagine.py --generator {VIDEO_GENERATOR} --k 1 --limit 1 --size {VIDEO_SIZE} --n-frames {N_FRAMES} --keep-videos 4 --out /content/smoke_video 2>&1 | tail -4
    from IPython.display import display, Video
    from PIL import Image
    import glob
    for f in sorted(glob.glob("/content/smoke_video/*/*.mp4")):
        print(f); display(Video(f, embed=True, width=256))
    for f in sorted(glob.glob("/content/smoke_video/*/*.png")):
        print(f); display(Image.open(f).resize((160, 160)))
else:
    print("skipped")
"""),
    code("""#@title 4. Video imagination for all trials (long)
%cd /content/world-maps
limit = f"--limit {IMAGINE_LIMIT}" if IMAGINE_LIMIT else ""
dirs = " ".join(DIRECTIONS)
!PYTHONPATH=src python scripts/cloth_imagine.py --generator {VIDEO_GENERATOR} --directions {dirs} --k {K_SAMPLES} --size {VIDEO_SIZE} --n-frames {N_FRAMES} {limit} 2>&1 | tail -20
import glob, os, numpy as np
from PIL import Image
from IPython.display import display
rows = []
for t in [3, 9, 15, 27]:
    d = "data/cloth/stimuli/unoccluded-occluded"
    ims = [f"{d}/trial_{t}_gtCloth.png", f"{d}/trial_{t}_gtNoCloth.png", f"{d}/trial_{t}_disNoCloth0.png",
           f"results/cloth/imagined/drape_video/trial_{t}_match_0.png", f"results/cloth/imagined/drape_video/trial_{t}_distractor_0.png",
           f"results/cloth/imagined/undrape_video/trial_{t}_target_0.png"]
    ims = [np.array(Image.open(p).convert("RGB").resize((160, 160))) if os.path.exists(p) else np.zeros((160, 160, 3), np.uint8) for p in ims]
    rows.append(np.concatenate(ims, 1))
sheet = Image.fromarray(np.concatenate(rows, 0)); sheet.save("results/cloth/imagined_video_contact_sheet.png"); display(sheet)
print("columns: target | match | distractor | video-draped match | video-draped distractor | video-undraped target")
"""),
    code("""#@title 5. Embed stimuli and imagined frames, then analyse
%cd /content/world-maps
args = " ".join(MODELS)
!PYTHONPATH=src python scripts/cloth_extract.py --models {args} 2>&1 | grep -v -i "warn\\|Loading weights\\|Fetching" | tail -8
!PYTHONPATH=src python scripts/cloth_analyze.py --n-boot 2000
from IPython.display import Image as IM, display
display(IM(filename="results/cloth/figures/trial_correlation_occluded_same.png"))
"""),
    code("""#@title 6. Package results for download
%cd /content/world-maps
!zip -q -r /content/world_maps_cloth_video_results.zip results/cloth/cloth_results.json results/cloth/cloth_results.md results/cloth/figures results/cloth/features/models.json results/cloth/imagined_video_contact_sheet.png $(ls results/cloth/imagined/*/*.mp4 2>/dev/null | head -12) 2>/dev/null
!ls -la /content/world_maps_cloth_video_results.zip
from google.colab import files
files.download("/content/world_maps_cloth_video_results.zip")
"""),
    md("""### If something fails
- **LTX-Video shape errors:** `VIDEO_SIZE` must be a multiple of 32 and `N_FRAMES` = 8k + 1 (49, 57, 65). For Wan use 4k + 1 (49, 81).
- **Out of memory (Wan):** lower `VIDEO_SIZE` to 480 or use `ltx`.
- **Quality check first:** run with `IMAGINE_LIMIT = 12` and look at the contact sheet and mp4s before the full run.
- **Runtime disconnects:** rerun cells 2 and 4; finished clips are skipped.
"""),
]
nb3 = {"cells": cells3, "metadata": META, "nbformat": 4, "nbformat_minor": 5}
out3 = REPO / "notebooks" / "colab_video_imagination.ipynb"
out3.write_text(json.dumps(nb3, indent=1))
print("wrote", out3)
h3 = HANDOFF.parent / "03_video_imagination.ipynb"
if HANDOFF.parent.exists():
    h3.write_text(json.dumps(nb3, indent=1))
    print("wrote", h3)

# ------------------------------------------------------------------ notebook 04: Physion, generators as simulators
PHYSION_FILES = {
    "src/world_maps/physion.py": REPO / "src/world_maps/physion.py",
    "scripts/physion_extract.py": REPO / "scripts/physion_extract.py",
    "scripts/physion_generate.py": REPO / "scripts/physion_generate.py",
    "scripts/physion_analyze.py": REPO / "scripts/physion_analyze.py",
}
cells4 = [
    md("""# World Maps — generators as simulators: do imagined outcomes predict what people predict? (Colab)

Physion (Bear et al. 2021): 1,144 test clips with per-stimulus human predictions (66–96 participants each) and ground
truth. People saw the first **1.5 s** and answered "will the red object touch the yellow area?". Here a video generator
**continues the 1.5 s context** and a validated red–yellow contact detector reads the imagined outcome. The question is
not only accuracy: do the generator's imagined outcomes agree with people on *which* stimuli are hard (partial
correlation after the label is removed), more than frozen-encoder readouts and a kinematic heuristic do?

| Stage | What | Time (A100) |
|---|---|---|
| Data + detector | 270 MB from S3, labels and human data from GitHub; detector validated on the real outcomes (CPU) | ≈ 8 min |
| Encoders | V-JEPA 2 L/H, VideoMAE-L, DINOv2-L, SigLIP 2 on the 1.5 s context (1,144 clips) | ≈ 15 min |
| VLM | Qwen3-VL P(yes) | ≈ 20 min |
| Generation | LTX-Video continuations for a stratified subset (default 40 per scenario = 320 clips, 1 sample) | ≈ 2–3 h |
| Analysis | humans, detector gate, readouts, generator, paired comparisons, figures | ≈ 5 min |

The last cell downloads `world_maps_physion_results.zip`. Nothing is pushed to GitHub.
"""),
    code("""#@title 1. Configuration
MODELS = ["vjepaL", "vjepaH", "vmaeL", "dinoL", "siglip2"]  #@param {type:"raw"}
RUN_VLM = True              #@param {type:"boolean"}
RUN_GENERATION = True       #@param {type:"boolean"}
GENERATOR = "ltx"           #@param ["ltx", "cosmos"]
PER_SCENARIO = 40           #@param {type:"integer"}   # stimuli per scenario (balanced on the label); 8 scenarios
K_SAMPLES = 1               #@param {type:"integer"}
GEN_SIZE = 512              #@param {type:"integer"}   # multiple of 32
N_FRAMES = 97               #@param {type:"integer"}   # 8k+1 for LTX
SMOKE_TEST_FIRST = True     #@param {type:"boolean"}
MOUNT_DRIVE = False         #@param {type:"boolean"}

import os, torch
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (switch to a GPU runtime)'}  {gpu_gb:.0f} GB")
VLM_ID = "Qwen/Qwen3-VL-8B-Instruct" if gpu_gb >= 20 else "Qwen/Qwen3-VL-4B-Instruct"
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    if tok: os.environ["HF_TOKEN"] = tok; print("HF_TOKEN loaded from Colab secrets")
except Exception:
    pass
"""),
    code("""#@title 2. Code and dependencies (about 2 min)
%cd /content
import os
if not os.path.exists("world-maps"):
    !git clone -q https://github.com/physicalreasoning/world-maps.git
%cd /content/world-maps
!git pull -q; git rev-parse --short HEAD
!pip install -q "transformers>=5.0" accelerate "diffusers>=0.33" imageio imageio-ffmpeg matplotlib pillow opencv-python-headless scikit-learn scipy pandas safetensors
import torch, transformers, diffusers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| diffusers", diffusers.__version__)
os.environ["PYTHONPATH"] = "/content/world-maps/src"
os.makedirs("results/physion", exist_ok=True)
"""),
]
for rel, path in PHYSION_FILES.items():
    cells4.append(code(f"%%writefile /content/world-maps/{rel}\n" + path.read_text()))
cells4 += [
    code("""#@title 3. Data (270 MB) and detector validation on the real outcomes (CPU; skipped when the committed results exist)
%cd /content/world-maps
!PYTHONPATH=src python scripts/physion_extract.py --stages detector 2>&1 | tail -14
"""),
    code("""#@title 4. Smoke test: one generated continuation, shown
%cd /content/world-maps
if SMOKE_TEST_FIRST and RUN_GENERATION:
    !PYTHONPATH=src python scripts/physion_generate.py --generator {GENERATOR} --per-scenario 2 --k 1 --size {GEN_SIZE} --n-frames {N_FRAMES} --limit 1 --out /content/smoke_gen 2>&1 | tail -5
    import glob, pandas as pd
    from IPython.display import display, Video
    for f in sorted(glob.glob("/content/smoke_gen/*/*.mp4"))[:1]:
        print(f); display(Video(f, embed=True, width=320))
    for f in glob.glob("/content/smoke_gen/*/scores.csv"):
        display(pd.read_csv(f).T)
else:
    print("skipped")
"""),
    code("""#@title 5. Encoders on the 1.5 s context (about 15 min) and VLM (optional, about 20 min)
%cd /content/world-maps
args = " ".join(MODELS)
!CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src python scripts/physion_extract.py --stages encoders --models {args} 2>&1 | grep -v -i "warn\\|Loading weights\\|Fetching\\|UNEXPECTED" | tail -8
if RUN_VLM:
    !PYTHONPATH=src python scripts/physion_extract.py --stages vlm --vlm-id {VLM_ID} 2>&1 | grep -v -i "warn\\|Loading weights\\|Fetching" | tail -3
"""),
    code("""#@title 6. Generation for the stratified subset (long; resumable)
%cd /content/world-maps
if RUN_GENERATION:
    !PYTHONPATH=src python scripts/physion_generate.py --generator {GENERATOR} --per-scenario {PER_SCENARIO} --k {K_SAMPLES} --size {GEN_SIZE} --n-frames {N_FRAMES} 2>&1 | tail -12
    import glob, numpy as np
    from PIL import Image
    from IPython.display import display
    frames = sorted(glob.glob(f"results/physion/generated/{GENERATOR}/frames/*_last.png"))[:12]
    if frames:
        sheet = np.concatenate([np.array(Image.open(f)) for f in frames], 1)
        Image.fromarray(sheet).save(f"results/physion/generated/{GENERATOR}/last_frames_sheet.png"); display(Image.fromarray(sheet))
else:
    print("skipped")
"""),
    code("""#@title 7. Analysis (CPU, about 5 min)
%cd /content/world-maps
!PYTHONPATH=src python scripts/physion_analyze.py --n-boot 1000
from IPython.display import Image as IM, display
for f in ["error_agreement.png", "accuracy_by_scenario.png"]:
    display(IM(filename=f"results/physion/figures/{f}"))
"""),
    code("""#@title 8. Package results for download
%cd /content/world-maps
!zip -q -r /content/world_maps_physion_results.zip results/physion/physion_results.json results/physion/physion_results.md results/physion/figures results/physion/detector_real.csv results/physion/kinematics.csv results/physion/stimuli.csv results/physion/vlm.csv results/physion/features/models.json $(ls results/physion/generated/*/scores.csv results/physion/generated/*/last_frames_sheet.png 2>/dev/null) $(ls results/physion/generated/*/*.mp4 2>/dev/null | head -8) 2>/dev/null
!ls -la /content/world_maps_physion_results.zip
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    !mkdir -p /content/drive/MyDrive/world_maps_physion && cp -r results/physion/features results/physion/generated /content/drive/MyDrive/world_maps_physion/ 2>/dev/null; ls /content/drive/MyDrive/world_maps_physion
from google.colab import files
files.download("/content/world_maps_physion_results.zip")
"""),
    md("""### If something fails
- **LTX conditioning errors:** `GEN_SIZE` must be a multiple of 32 and `N_FRAMES` = 8k + 1 (97, 121). The context is resampled to 24 fps (36 frames → 33 used).
- **Cosmos:** experimental; needs the `cosmos_guardrail` package and Hub access. Use `ltx` if it fails.
- **Generation too slow:** lower `PER_SCENARIO` to 24 (192 clips) or `N_FRAMES` to 73; the analysis uses whatever exists.
- **Runtime disconnects:** rerun cells 2 and 6; finished clips are skipped (scores.csv is appended).
"""),
]
nb4 = {"cells": cells4, "metadata": META, "nbformat": 4, "nbformat_minor": 5}
out4 = REPO / "notebooks" / "colab_physion_simulators.ipynb"
out4.write_text(json.dumps(nb4, indent=1))
print("wrote", out4)
h4 = HANDOFF.parent / "04_physion_simulators.ipynb"
if HANDOFF.parent.exists():
    h4.write_text(json.dumps(nb4, indent=1))
    print("wrote", h4)
