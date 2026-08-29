# Machine setup runbook — CADC weather-robustness project

**Purpose.** Rebuild this project's working environment on a fresh machine from a bare
git clone. Every command and version here was executed and verified on the original box
(Tesla T4, Ubuntu 22.04, driver 580.173.02 / CUDA 13.0, nvcc 12.9) on 2026-08-29.

**How to use.** Hand this file to Claude Code in a fresh clone and say "follow SETUP.md".
Or run the steps yourself, in order. `scripts/setup_env.sh` automates steps 1–5.

Each step ends with a **Verify** block. Do not proceed past a failing verify — every
later failure in this stack is caused by an earlier step silently not working.

---

## 0. Machine requirements

| | |
|---|---|
| GPU | Any CUDA GPU. Original: Tesla T4 (15 GB) |
| Driver | Must support CUDA 12.4 runtime. 580.x is fine |
| Disk | See step 6. `--parts lidar` ≈ **10 GiB**; `--parts full` ≈ **195 GiB**. Add headroom for `gt_database/`, snow-sim copies and checkpoints — 500 GB is comfortable |
| OS | Ubuntu 22.04. Needs `sudo` for `build-essential` |

> **Disk note.** 94% of CADC is camera imagery this project never reads. The only image
> use in the whole loader is `get_image_shape()`. Budget for the mode you pick in step 6,
> plus room for `gt_database/`, snow-sim augmented copies, and checkpoints.

---

## 1. System packages

OpenPCDet compiles custom C++/CUDA kernels; without a compiler the build fails with
`c++: not found` / `cannot execute 'cc1plus'` / `nvcc fatal: Failed to preprocess host
compiler properties`.

```bash
sudo apt-get update
sudo apt-get install -y build-essential zstd
```

**Verify:** `g++ --version && nvcc --version && zstd --version` all print versions.
If `nvcc` is missing, install the CUDA toolkit — `nvidia-smi` alone is not enough.

---

## 2. Conda environment

Python **3.10**. Not 3.11+, and definitely not 3.13: prebuilt `spconv` wheels do not
exist for 3.13, which is what `ERROR: Could not find a version that satisfies the
requirement spconv-cu120` actually means.

```bash
conda create -n stress_test python=3.10 -y
conda activate stress_test
```

---

## 3. Python stack — install in this exact order

Order matters: OpenPCDet's `setup.py` imports `torch` at build time, so torch must
already be present.

```bash
# 3a. PyTorch. cu124 wheels work against any CUDA 12.x driver/toolkit (this box had 12.9).
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 3b. numpy MUST stay below 2.0 — numpy 2.x breaks binary compat with spconv/cumm's
#     C++ extensions, which surfaces as confusing ImportErrors rather than a clear message.
pip install "numpy<2.0"

# 3c. spconv — install ONLY the cu124 build.
#     Do not also install spconv-cu120: both provide the `spconv` package and overwrite
#     each other in site-packages, leaving a half-clobbered install.
pip install spconv-cu124==2.3.8

# 3d. Remaining OpenPCDet deps
pip install easydict==1.13 SharedArray tensorboardX tqdm pyyaml scikit-image \
            numba scipy protobuf ninja

# 3e. For the selective data downloader (step 6)
pip install remotezip
```

**Verify:**
```bash
python - <<'EOF'
import torch, numpy, spconv
from spconv.utils import Point2VoxelCPU3d
assert torch.cuda.is_available(), "CUDA not visible to torch"
assert numpy.__version__ < "2", f"numpy too new: {numpy.__version__}"
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| numpy", numpy.__version__, "| spconv", spconv.__version__)
EOF
```
Expected: `torch 2.6.0+cu124 | cuda 12.4 | numpy 1.26.4 | spconv 2.3.8`

> `from spconv.utils import VoxelGeneratorV2` will **always** fail on spconv 2.x —
> that class was deleted, not moved. See patch P3 in the appendix. Reinstalling spconv
> does not fix it and never will.

---

## 4. Build OpenPCDet

`OpenPCDet/` is vendored **in this repo** with all four compatibility patches already
applied (appendix). Do not re-clone `mpitropov/OpenPCDet` — you would lose them.

```bash
cd OpenPCDet
pip install -v -e . --no-build-isolation
cd ..
```

`--no-build-isolation` is required. Without it pip builds in an isolated env with no
torch and fails with `ModuleNotFoundError: No module named 'torch'`. `python setup.py
develop` is deprecated and routes to the same broken isolated build.

**Verify:**
```bash
python -c "
import pcdet; print('pcdet', pcdet.__version__)
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
print('CUDA ops OK')"
```
Expected: `pcdet 0.3.0+c2ef9b0` then `CUDA ops OK`.

---

## 5. PYTHONPATH

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/OpenPCDet"     # add to ~/.bashrc to persist
```

---

## 6. Download the data

**URLs — verified live. The `cadcd_data`/`cadc_clear_data` paths circulating elsewhere
are wrong; `cadc_clear_data` returns 404.**

| Dataset | URL | Dates | Archive |
|---|---|---|---|
| CADC (snowy) | `https://wiselab.uwaterloo.ca/cadc/` | 2018_03_06, 2018_03_07, 2019_02_27 | `labeled.zip` |
| CADC-clear | `https://wiselab.uwaterloo.ca/cadc-clear/` | 2018_02_22, 2018_03_05, 2018_03_21, 2018_04_02 | `labeled.tar.zst` |

75 sequences each. CADC = 89.6 GiB, CADC-clear = 105.6 GiB. Note the **different archive
formats** — a script written for CADC's `.zip` cannot read CADC-clear.

```bash
python scripts/fetch_cadc.py --parts lidar    # LiDAR + GPS + calib
python scripts/fetch_cadc.py --parts full     # adds all 8 cameras
python scripts/fetch_cadc.py --parts gps      # GPS only, for trajectory matching
```

| mode | transferred | on disk | contents |
|---|---|---|---|
| `gps` | ~4 GiB | 61 MB | per-frame GPS only |
| `lidar` | ~110 GiB | ~10 GiB | LiDAR + GPS + calib — everything this project reads |
| `full` | ~195 GiB | ~195 GiB | adds all 8 cameras |

`lidar` still transfers ~110 GiB because CADC-clear's `.tar.zst` is a single zstd stream
with no random access — all 105.6 GiB must flow past even though 94% is discarded. CADC's
`.zip` does support range requests, so its LiDAR costs only ~5 GiB. Disk is what
selective extraction saves on the clear half, not bandwidth.

Both modes are resumable — re-run after an interruption and completed sequences are
skipped. See `scripts/fetch_cadc.py --help`.

**Verify:** `python scripts/fetch_cadc.py --parts lidar --verify-only` reports
`0 mismatched` for both halves. CADC must total exactly **7,000 frames** (the published
figure); CADC-clear totals 7,873.

### If you chose `--parts lidar`

Apply patch **P5** (appendix) so `get_image_shape()` returns the constant `[1024, 1280]`
instead of opening a PNG that isn't there. Every CADC image is that size.

---

## 7. ImageSets

Frame-level splits, one line per frame as `DATE SEQ FRAME` (space-separated):

```
2019_02_27 0002 0000000000
```

The full 7,000-frame splits ship with the fork at `OpenPCDet/data/cadc/ImageSets/`
(5,600 train / 1,400 val) and are already in this repo. `data/cadc/ImageSets/` holds the
smoke-test subset (57/14 frames of `2019_02_27/0002`), also tracked.

To build a subset for one sequence:
```bash
grep "^2019_02_27 0002 " OpenPCDet/data/cadc/ImageSets/train.txt | sort -k3 > /tmp/seq.txt
head -57 /tmp/seq.txt > data/cadc/ImageSets/train.txt
tail -14 /tmp/seq.txt > data/cadc/ImageSets/val.txt
: > data/cadc/ImageSets/test.txt     # MUST exist, even empty
```

> `test.txt` must exist. `create_cadc_infos` unconditionally calls `set_split('test')`;
> a missing file leaves `sample_id_list = None` and `executor.map` crashes.

---

## 8. Generate infos

```bash
python make_cadc_infos.py
```

Writes `cadc_infos_{train,val,trainval,test}.pkl` and `cadc_dbinfos_train.pkl` into
`data/cadc/`, plus `gt_database/`.

Use this driver rather than `python -m pcdet.datasets.cadc.cadc_dataset create_cadc_infos
...` — that `__main__` block hardcodes `data_path` to `OpenPCDet/data/cadc` and reads
`DATA_PATH: '/root/cadc'` from the yaml, neither of which is right here.

---

## 9. Smoke test

```bash
python smoke_test_loader.py
```

Expected (eval and training modes, the latter exercising gt_sampling + augmentation +
voxelization):
```
=== VAL  (classes: ['Car', 'Pedestrian', 'Pickup_Truck']) ===
Dataset length   : 14 frames
points           : (39475, 4)  ...
voxels           : (34953, 5, 4)
gt_boxes         : (18, 8)  {'Car': 14, 'Pickup_Truck': 4}
```

Two non-obvious but **correct** behaviours:
- `z` exceeds `POINT_CLOUD_RANGE`'s `[-3, 3]`. `mask_points_by_range` filters x/y only;
  z is clipped inside the voxel generator.
- Training mode uses only `['Car', 'Pickup_Truck']`. Sequence `2019_02_27/0002` contains
  **no pedestrians**, and `database_sampler.py:23` does `db_infos[cur_class]` with no
  default, so it `KeyError`s on any class absent from the gt database. On a
  pedestrian-free subset, either restrict the class list or change that to
  `infos.get(cur_class, [])`.

---

## Appendix A — vendored patches

Already applied in this repo. Listed so they can be re-applied if OpenPCDet is ever
re-cloned from `mpitropov/OpenPCDet@cadc_support` (which targets spconv 1.x + torch 1.x).

**P1 — legacy THC headers (`pcdet/ops/**/*.cpp`).** PyTorch 2.x removed `THC/THC.h`;
build dies with `fatal error: THC/THC.h: No such file or directory`. Run
`python OpenPCDet/patch_openpcdet.py` from inside `OpenPCDet/`:
`#include <THC/THC.h>` → `<ATen/cuda/CUDAContext.h>`, `THCState_getCurrentStream(state)`
→ `at::cuda::getCurrentCUDAStream()`, `THCudaCheck` → `AT_CUDA_CHECK`, and the
`extern THCState *state;` declarations commented out.

**P2 — `yaml.load` without Loader (`cadc_dataset.py:550`).** PyYAML ≥5.1 requires an
explicit loader: `yaml.load(open(...))` → `yaml.safe_load(open(...))`.

**P3 — spconv 2.x voxel generator (`pcdet/datasets/processor/data_processor.py`).**
`VoxelGeneratorV2` does not exist in any spconv 2.x release. Replaced the dead
import chain with `VoxelGeneratorWrapper` (defined at the bottom of that file), backed by
`spconv.utils.Point2VoxelCPU3d` and built lazily on first call so `num_point_features`
can be read off the points without changing `DataProcessor.__init__`'s signature.

**P4 — global cfg reference (`cadc_dataset.py:465`).** `cfg.DATA_CONFIG.FOV_POINTS_ONLY`
→ `self.dataset_cfg.get('FOV_POINTS_ONLY', False)`. The global `cfg` has no `DATA_CONFIG`
when a dataset-only yaml is loaded. Correct either way, since `dataset_cfg` *is*
`cfg.DATA_CONFIG` when loaded via a model config.

**P5 — image-free operation (only if you skipped the cameras).** In
`cadc_dataset.get_image_shape()`, return `np.array([1024, 1280], dtype=np.int32)` instead
of `io.imread(img_file).shape[:2]`. Every CADC image is 1024×1280. Not applied by
default — apply it only when running `--parts lidar`.

---

## Appendix B — error → cause

| Error | Cause |
|---|---|
| `Could not find a version that satisfies the requirement spconv-cu120` | Python 3.11+/3.13. Use 3.10 (step 2) |
| `ModuleNotFoundError: No module named 'torch'` during install | Missing `--no-build-isolation` (step 4) |
| `c++: not found` / `cannot execute 'cc1plus'` | `build-essential` not installed (step 1) |
| `fatal error: THC/THC.h: No such file or directory` | Patch P1 not applied |
| `TypeError: load() missing 1 required positional argument: 'Loader'` | Patch P2 not applied |
| `cannot import name 'VoxelGeneratorV2' from 'spconv.utils'` | Patch P3 not applied. **Reinstalling spconv will not help** |
| `AttributeError: 'EasyDict' object has no attribute 'DATA_CONFIG'` | Patch P4, or a script reading `cfg.DATA_CONFIG` from a dataset-only yaml |
| `No such file or directory: '.../cadc_dbinfos_train.pkl'` | Step 8 not run |
| `KeyError: 'Pedestrian'` in `database_sampler.py` | Class absent from the gt database — see step 9 |
| `ValueError: not enough values to unpack` from `get_lidar` | ImageSets rows are not `DATE SEQ FRAME` (step 7) |
| Downloader creates `data/cadc/data/cadc/cadcd/` | The devkit's `download_cadcd.py` `os.chdir`s then re-uses a *relative* path. Use `scripts/fetch_cadc.py` |

---

## Appendix C — open issues carried over

1. **The CADC+ pairing table is not published.** Verified absent from both file servers,
   the per-sequence `3d_ann.json`, `cadc_devkit`, `github.com/wiselabuw`, the IV 2025
   paper, and the MASc thesis. The Segments.ai public API returns split membership but
   empty `metadata`. `EMAIL_DRAFT.md` is a ready request to the authors.

   Thesis §4.5: only **53 of 74** pairs are genuine same-road spatial matches. 15 were
   matched manually to a *different* but similar scene, 6 on road-agent type alone. This
   matters — the Level 2 axis assumes location is held constant, and for 28% of pairs it
   is not. Stratify results by match quality.

   `scripts/fetch_cadc.py --parts gps` retrieves per-frame GPS for all 150 sequences
   (~61 MB) — enough to reconstruct the 53 spatial matches. The 21 manual ones cannot be
   derived.

2. **CADC-clear's training set is sparsely labelled — every 10th frame only.** Validation
   is dense. `make_cadc_infos.py` currently assumes dense labels.

3. **`LiDAR_snow_sim` is not in this repo** despite commit `3bfbfd8`'s message. Gate 400
   needs it re-cloned, plus recalibration from the HDL-64E beam model to CADC's 32-beam
   VLP-32C.

4. **Split membership** (from the Segments.ai public API, no auth required):
   CADC 60 train + 14 val = 74; CADC-clear 60 train + 15 val = 75. The 14-vs-15 gap is
   the one CADC sequence matched to two half-length clear sequences.
