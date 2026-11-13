// Test-only CUDA extension: runs cuda_rasterizer/texture_sampler.h on a batch of samples,
// so the sampler can be checked against PyTorch before it goes into the rasterizer (step 2).
// Built on the fly by tests/test_texture_sampler.py (torch.utils.cpp_extension.load).
#include <torch/extension.h>
#include "texture_sampler.h"

// sample n: texture tex_id[n], coordinates uv[n] -> out[n, 0..C-1]
__global__ void forward_kernel(int N, const float* uv, const int* tex_id, const float* tex,
                               int S, int C, float* out)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float r[16];
    sample_texture_bilinear(r, uv[2 * n], uv[2 * n + 1], tex + (size_t)tex_id[n] * C * S * S, S, C);
    for (int c = 0; c < C; ++c) out[n * C + c] = r[c];
}

__global__ void backward_kernel(int N, const float* uv, const int* tex_id, const float* tex,
                                const float* dout, int S, int C, float* dtex, float* duv)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    size_t off = (size_t)tex_id[n] * C * S * S;
    float2 g = backward_sample_texture_bilinear(dout + n * C, uv[2 * n], uv[2 * n + 1],
                                                tex + off, dtex + off, S, C);
    duv[2 * n] = g.x;
    duv[2 * n + 1] = g.y;
}

// tex: [T, C, S, S] float32, uv: [N, 2] float32, tex_id: [N] int32  ->  [N, C]
torch::Tensor forward(torch::Tensor tex, torch::Tensor uv, torch::Tensor tex_id)
{
    const int N = uv.size(0), C = tex.size(1), S = tex.size(2);
    TORCH_CHECK(C <= 16 && S >= 2 && tex.size(3) == S, "bad texture shape");
    auto out = torch::empty({N, C}, tex.options());
    forward_kernel<<<(N + 255) / 256, 256>>>(N, uv.data_ptr<float>(), tex_id.data_ptr<int>(),
                                             tex.data_ptr<float>(), S, C, out.data_ptr<float>());
    return out;
}

// returns (dL/dtex [T, C, S, S], dL/duv [N, 2])
std::vector<torch::Tensor> backward(torch::Tensor tex, torch::Tensor uv, torch::Tensor tex_id,
                                    torch::Tensor dout)
{
    const int N = uv.size(0), C = tex.size(1), S = tex.size(2);
    auto dtex = torch::zeros_like(tex);
    auto duv = torch::empty({N, 2}, uv.options());
    backward_kernel<<<(N + 255) / 256, 256>>>(N, uv.data_ptr<float>(), tex_id.data_ptr<int>(),
                                              tex.data_ptr<float>(), dout.data_ptr<float>(), S, C,
                                              dtex.data_ptr<float>(), duv.data_ptr<float>());
    return {dtex, duv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward);
    m.def("backward", &backward);
}
