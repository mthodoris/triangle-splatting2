# =============================================================================
# Image for textured-triangles (combined/) on the lab cluster.
# Runs on RTX 3090 (sm_86), 4090 (sm_89) and 5090 (sm_120).
#
# Same package versions as the local combined/.venv (notes/decisions.md), except:
#   - torch/torchvision/torchaudio use the CUDA 12.8 build (+cu128) instead of +cu126,
#     because cu126 has no RTX 5090 support. The nvidia-*/triton pins in
#     requirements-venv.txt are for cu126, so they are skipped; torch brings its own.
#
# Base is plain ubuntu:24.04, NOT nvidia/cuda: the nvidia/cuda images bake in
# NVIDIA_VISIBLE_DEVICES=all, which breaks GPU isolation and is rejected by
# ~/cluster-management/scripts/apply-job.sh (docs/admin/GPU-ISOLATION-FIX.md).
# Only the CUDA 12.8 toolkit (nvcc + headers, no driver) is installed, to build our
# extensions; the CUDA runtime libraries come with the torch wheels.
#
# The code is cloned from the fork at a fixed commit (no COPY), so the build
# context is empty (.dockerignore excludes everything). Build/push commands:
# notes/cluster.md.
# =============================================================================
FROM ubuntu:24.04

ARG REPO=https://github.com/mthodoris/triangle-splatting2.git
ARG COMMIT
ARG CUDA_ARCHS="8.6;8.9;12.0+PTX"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential ninja-build pkg-config time \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# --- CUDA 12.8 toolkit only (no driver), from NVIDIA's apt repository ---
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-8 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=/usr/local/cuda-12.8/bin:$PATH
# Needed by the NVIDIA container runtime. NVIDIA_VISIBLE_DEVICES is set per job by Kubernetes.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYOPENGL_PLATFORM=egl

# --- Python 3.10.12 via uv (uv only installs Python), then a normal venv ---
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/uv
RUN UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_PYTHON_BIN_DIR=/opt/uv-bin uv python install 3.10.12 \
    && /opt/uv-python/cpython-3.10.12-linux-x86_64-gnu/bin/python3.10 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir pip==22.0.2 setuptools==59.6.0

# --- code at a fixed commit, submodules at their pinned commits (no --remote) ---
RUN test -n "${COMMIT}" || (echo "build with --build-arg COMMIT=<git sha>" && exit 1)
RUN git clone ${REPO} /app/combined \
    && cd /app/combined && git checkout --detach ${COMMIT} \
    && git submodule update --init --recursive \
    && git rev-parse HEAD > /app/IMAGE_COMMIT
WORKDIR /app/combined

# --- packages: torch (cu128), then the pinned list minus cu126-specific nvidia/triton pins ---
RUN pip install --no-cache-dir torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
RUN grep -vE '^(nvidia-|triton==)' requirements-venv.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && pip install --no-cache-dir tbb-devel==2021.11.0 tbb==2021.11.0

# --- CUDA extensions (arch list inline, as the lab template requires) ---
RUN TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" FORCE_CUDA=1 MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@5043d15361d16a7093b4b60572c5f730c6c83308"
RUN TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" MMCV_WITH_OPS=1 MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation \
        --no-binary mmcv mmcv==2.2.0
RUN TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn \
    && TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" pip install --no-cache-dir --no-build-isolation ./submodules/diff-triangle2-rasterization
RUN cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/app/combined/triangulation \
        -DPython3_EXECUTABLE=/opt/venv/bin/python \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
        -DTBB_DIR=/opt/venv/lib/cmake/tbb \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share/cmake"))')" \
        -DCMAKE_CUDA_ARCHITECTURES="86;89;120" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    && cmake --build build -j8 && cmake --install build

# --- download runtime models into the image (jobs need no internet, all jobs use the same files) ---
ENV TORCH_HOME=/opt/torch_home
RUN python -c "import torch; torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)" \
    && python -c "import lpips; lpips.LPIPS(net='vgg')" \
    && python -c "from lpipsPyTorch import lpips; import torch; x=torch.rand(1,3,64,64); print(lpips(x,x,net_type='vgg'))" \
    && python -c "import torchvision; torchvision.models.vgg16(weights='IMAGENET1K_V1')"

# --- checks ---
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list())" \
    && python -c "import diff_textured_triangle_rasterization, simple_knn._C, pytorch3d, mmcv.ops, triangulation; print('imports ok')" \
    && (env | grep -E '^(NVIDIA_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES)=' && echo 'GPU env var baked in!' && exit 1 || true)

CMD ["/bin/bash"]
