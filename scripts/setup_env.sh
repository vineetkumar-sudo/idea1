#!/usr/bin/env bash
# Rebuild the conda environment and build OpenPCDet. Steps 1-5 of SETUP.md.
#
#   bash scripts/setup_env.sh [env_name]     # default env name: stress_test
#
# Idempotent: safe to re-run. Does not download data (see scripts/fetch_cadc.py).
set -euo pipefail

ENV_NAME="${1:-stress_test}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. System packages"
if ! command -v g++ >/dev/null || ! command -v zstd >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y build-essential zstd
fi
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found. Install the CUDA toolkit (nvidia-smi alone is not enough)."; exit 1; }
g++ --version | head -1
nvcc --version | tail -2 | head -1

say "2. Conda env '$ENV_NAME' (Python 3.10)"
# 3.11+ has no prebuilt spconv wheels; 3.13 is what caused
# "Could not find a version that satisfies the requirement spconv-cu120".
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE "^${ENV_NAME}\s" || conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"
python -V

say "3. Python stack (order matters -- OpenPCDet's setup.py imports torch at build time)"
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q "numpy<2.0"                 # numpy 2.x breaks spconv/cumm binary compat
pip uninstall -y -q spconv-cu120 cumm-cu120 2>/dev/null || true   # only one spconv build may be installed
pip install -q spconv-cu124==2.3.8
pip install -q easydict==1.13 SharedArray tensorboardX tqdm pyyaml scikit-image \
               numba scipy protobuf ninja remotezip

say "3b. Verify the stack"
python - <<'EOF'
import torch, numpy, spconv
from spconv.utils import Point2VoxelCPU3d
assert torch.cuda.is_available(), "CUDA not visible to torch"
assert numpy.__version__ < "2", f"numpy too new: {numpy.__version__}"
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"numpy {numpy.__version__} | spconv {spconv.__version__}")
EOF

say "4. Build OpenPCDet (vendored, patches already applied -- do NOT re-clone)"
# --no-build-isolation is required: without it pip builds in an isolated env with no
# torch and fails with "ModuleNotFoundError: No module named 'torch'".
( cd OpenPCDet && pip install -v -e . --no-build-isolation 2>&1 | tail -5 )

say "5. Verify pcdet + compiled CUDA ops"
export PYTHONPATH="${PYTHONPATH:-}:$REPO/OpenPCDet"
python - <<'EOF'
import pcdet
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
print("pcdet", pcdet.__version__, "| CUDA ops OK | spconv 2.x shim OK")
EOF

cat <<EOF

Environment ready.

  conda activate $ENV_NAME
  export PYTHONPATH="\$PYTHONPATH:$REPO/OpenPCDet"

Next: step 6 of SETUP.md
  python scripts/fetch_cadc.py --parts lidar
  python make_cadc_infos.py
  python smoke_test_loader.py
EOF
