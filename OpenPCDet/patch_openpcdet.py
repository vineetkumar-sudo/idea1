import os
import glob

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # 1. Replace the missing header
    new_content = content.replace('#include <THC/THC.h>', '#include <ATen/cuda/CUDAContext.h>')
    
    # 2. Remove the legacy THCState state declarations
    new_content = new_content.replace('extern THCState *state;', '// extern THCState *state;')
    new_content = new_content.replace(
        'THCState *state = at::globalContext().lazyInitCUDA();', 
        '// THCState *state = at::globalContext().lazyInitCUDA();'
    )
    
    # 3. Replace the stream getter (THCState_getCurrentStream(state) -> at::cuda::getCurrentCUDAStream())
    # Note: We use .stream() to get the raw cudaStream_t if the code expects it
    new_content = new_content.replace(
        'THCState_getCurrentStream(state)', 
        'at::cuda::getCurrentCUDAStream()'
    )
    
    # 4. Replace the error checker
    new_content = new_content.replace('THCudaCheck', 'AT_CUDA_CHECK')
    
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"✅ Patched: {path}")

# Find all C++ and CUDA files in the ops directory
ops_dir = 'pcdet/ops'
files = glob.glob(os.path.join(ops_dir, '**/*.cpp'), recursive=True) + \
        glob.glob(os.path.join(ops_dir, '**/*.cu'), recursive=True)

for f in files:
    patch_file(f)