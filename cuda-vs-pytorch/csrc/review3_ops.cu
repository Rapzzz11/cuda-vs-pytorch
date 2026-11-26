#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#define CHECK_INPUT(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kFloat32, #x " must be a float32 CUDA tensor")
#define CHECK_LAUNCH() TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed")

static int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

// ---------------------------------------------------------------- ReLU
__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) { float v = x[i]; y[i] = v > 0.0f ? v : 0.0f; }
}
torch::Tensor relu_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto xc = x.contiguous();
    auto y = torch::empty_like(xc);
    int n = xc.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    relu_kernel<<<blocks, threads>>>(xc.data_ptr<float>(), y.data_ptr<float>(), n);
    CHECK_LAUNCH();
    return y;
}

// ------------------------------------------- MatMul v1: shared-memory tiling
#define TILE 16
__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx, bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE; ++i) acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    auto Ac = A.contiguous(), Bc = B.contiguous();
    int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    TORCH_CHECK(Bc.size(0) == K, "inner dimensions must match");
    auto C = torch::empty({M, N}, Ac.options());
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(Ac.data_ptr<float>(), Bc.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    CHECK_LAUNCH();
    return C;
}

// ------------------------- MatMul v2: shared tiling + register blocking (4x4 per thread)
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
__global__ void matmul_reg_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;       // 16 x 16 threads
    int tid = ty * 16 + tx;
    int rowBase = blockIdx.y * BM, colBase = blockIdx.x * BN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {              // A tile: BM x BK = 1024 elements
            int idx = tid + i * 256, r = idx / BK, c = idx % BK;
            int gr = rowBase + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {              // B tile: BK x BN = 1024 elements
            int idx = tid + i * 256, r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = colBase + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = rowBase + ty * TM + i, c = colBase + tx * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}
torch::Tensor matmul_reg_forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    auto Ac = A.contiguous(), Bc = B.contiguous();
    int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    TORCH_CHECK(Bc.size(0) == K, "inner dimensions must match");
    auto C = torch::empty({M, N}, Ac.options());
    dim3 block(16, 16), grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_reg_kernel<<<grid, block>>>(Ac.data_ptr<float>(), Bc.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    CHECK_LAUNCH();
    return C;
}

// ------------------------------------------------------- Softmax (row-wise)
__global__ void softmax_kernel(const float* __restrict__ X, float* __restrict__ Y, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float* xr = X + (size_t)row * cols;
    float* yr = Y + (size_t)row * cols;
    float m = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) m = fmaxf(m, xr[i]);
    sdata[tid] = m; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]); __syncthreads(); }
    float row_max = sdata[0]; __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) { float e = expf(xr[i] - row_max); yr[i] = e; sum += e; }
    sdata[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float row_sum = sdata[0];
    for (int i = tid; i < cols; i += blockDim.x) yr[i] = yr[i] / row_sum;
}
torch::Tensor softmax_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto xc = x.contiguous();
    TORCH_CHECK(xc.dim() == 2, "softmax expects a 2D tensor [rows, cols]");
    int rows = xc.size(0), cols = xc.size(1);
    auto y = torch::empty_like(xc);
    int threads = next_pow2(cols < 1024 ? cols : 1024);
    if (threads < 32) threads = 32;
    softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(xc.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    CHECK_LAUNCH();
    return y;
}

// ---------------------------------------------------------------- LayerNorm
__global__ void layernorm_kernel(const float* __restrict__ X, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, float* __restrict__ Y,
                                 int rows, int cols, float eps) {
    extern __shared__ float sdata[];
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float* xr = X + (size_t)row * cols;
    float* yr = Y + (size_t)row * cols;
    float s = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) s += xr[i];
    sdata[tid] = s; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (tid < k) sdata[tid] += sdata[tid + k]; __syncthreads(); }
    float mean = sdata[0] / cols; __syncthreads();
    float v = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) { float d = xr[i] - mean; v += d * d; }
    sdata[tid] = v; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (tid < k) sdata[tid] += sdata[tid + k]; __syncthreads(); }
    float inv_std = rsqrtf(sdata[0] / cols + eps);
    for (int i = tid; i < cols; i += blockDim.x) yr[i] = (xr[i] - mean) * inv_std * gamma[i] + beta[i];
}
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(x); CHECK_INPUT(gamma); CHECK_INPUT(beta);
    auto xc = x.contiguous(), gc = gamma.contiguous(), bc = beta.contiguous();
    TORCH_CHECK(xc.dim() == 2, "layernorm expects a 2D tensor [rows, cols]");
    int rows = xc.size(0), cols = xc.size(1);
    auto y = torch::empty_like(xc);
    int threads = next_pow2(cols < 1024 ? cols : 1024);
    if (threads < 32) threads = 32;
    layernorm_kernel<<<rows, threads, threads * sizeof(float)>>>(xc.data_ptr<float>(), gc.data_ptr<float>(),
                                                                 bc.data_ptr<float>(), y.data_ptr<float>(), rows, cols, (float)eps);
    CHECK_LAUNCH();
    return y;
}

// ------------------------------------------- Conv2d (direct, NCHW, naive)
__global__ void conv2d_kernel(const float* __restrict__ in, const float* __restrict__ w,
                              const float* __restrict__ bias, float* __restrict__ out,
                              int N, int Cin, int H, int W, int Cout, int KH, int KW,
                              int OH, int OW, int stride, int pad) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z, n = nc / Cout, co = nc % Cout;
    if (ow >= OW || oh >= OH || n >= N) return;
    float acc = 0.0f;
    for (int ci = 0; ci < Cin; ++ci)
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride - pad + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride - pad + kw;
                if (iw < 0 || iw >= W) continue;
                acc += in[((n * Cin + ci) * H + ih) * W + iw] * w[((co * Cin + ci) * KH + kh) * KW + kw];
            }
        }
    out[((n * Cout + co) * OH + oh) * OW + ow] = acc + bias[co];
}
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t pad) {
    CHECK_INPUT(input); CHECK_INPUT(weight); CHECK_INPUT(bias);
    auto in = input.contiguous(), w = weight.contiguous(), b = bias.contiguous();
    int N = in.size(0), Cin = in.size(1), H = in.size(2), W = in.size(3);
    int Cout = w.size(0), KH = w.size(2), KW = w.size(3);
    TORCH_CHECK(w.size(1) == Cin, "channel mismatch");
    int OH = (H + 2 * pad - KH) / stride + 1, OW = (W + 2 * pad - KW) / stride + 1;
    auto out = torch::empty({N, Cout, OH, OW}, in.options());
    dim3 block(16, 16, 1), grid((OW + 15) / 16, (OH + 15) / 16, N * Cout);
    conv2d_kernel<<<grid, block>>>(in.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                                   N, Cin, H, W, Cout, KH, KW, OH, OW, (int)stride, (int)pad);
    CHECK_LAUNCH();
    return out;
}
