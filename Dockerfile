FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0+PTX"
ENV CMAKE_CUDA_ARCHITECTURES="86;89;90;120"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git build-essential cmake ninja-build pkg-config \
    libgmp3-dev libmpfr-dev libboost-all-dev libglm-dev libtbb-dev \
    libeigen3-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libglvnd0 libglx0 libegl1 libgles2 \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.11 -m venv /app/.venv

RUN pip install --upgrade pip "setuptools<81" wheel packaging pybind11 ninja cmake

RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN echo "$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf && ldconfig

ARG CACHEBUST=1
RUN git clone --recursive -b devel https://github.com/mthodoris/triangle-splatting2.git

WORKDIR /app/triangle-splatting2

RUN git submodule update --init --recursive --remote

RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# --- Build diff-triangle2-rasterization CUDA extension ---
RUN cd submodules/diff-triangle2-rasterization \
    && rm -rf build dist diff_triangle_rasterization.egg-info \
    && pip install --no-cache-dir --no-build-isolation .

# --- Build simple-knn CUDA extension ---
RUN cd submodules/simple-knn && pip install --no-cache-dir --no-build-isolation .

# --- Build Delaunay triangulation module (adapted from RadFoam) ---
RUN cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$(pwd)/triangulation" \
    -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    && cmake --build build -j"${MAX_JOBS}" \
    && cmake --install build

RUN python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
RUN python -c "import diff_triangle_rasterization; print('diff_triangle_rasterization ok')"
RUN python -c "import simple_knn; print('simple_knn ok')"

ENTRYPOINT ["bash", "-c", "git pull origin devel && exec \"$@\"", "--"]
WORKDIR /app/triangle-splatting2

CMD ["/bin/bash"]
