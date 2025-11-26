// src/fft_convolution.cu
// GPU FFT-based 2D convolution (PNG in -> PNG out). Auto-pads to next pow2.
// Uses cuFFT (R2C / C2R) and CUDA kernels for pointwise multiply.
// Default kernel: 7x7 box blur (change `kernel_size` variable).
//
// Build with nvcc and link OpenCV + cuFFT (Makefile provided).

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace std;
using namespace cv;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) \
             << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult r = call; \
    if (r != CUFFT_SUCCESS) { \
        cerr << "CUFFT error at " << __FILE__ << ":" << __LINE__ << " code=" << r << endl; \
        exit(1); \
    } \
} while(0)

// pointwise complex multiply (A *= B)
__global__ void pointwiseMul(cufftComplex* A, const cufftComplex* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    cufftComplex a = A[idx];
    cufftComplex b = B[idx];
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    A[idx] = c;
}

// clamp and convert float -> uchar (performed after cropping)
inline unsigned char clamp255f(float v) {
    int iv = (int)roundf(v);
    if (iv < 0) iv = 0;
    if (iv > 255) iv = 255;
    return (unsigned char)iv;
}

// next power of two helper
static inline int nextPow2(int v) {
    if (v <= 1) return 1;
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

int main(int argc, char** argv) {
    string inPath = "data/input.png";
    string outPath = "output/output.png";
    int kernel_size = 7; // default 7x7 box blur

    if (argc >= 2) inPath = argv[1];
    if (argc >= 3) outPath = argv[2];
    if (argc >= 4) kernel_size = atoi(argv[3]);

    // Read input PNG (grayscale)
    Mat img = imread(inPath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: cannot open input image: " << inPath << endl;
        return 1;
    }
    int H = img.rows;
    int W = img.cols;
    cout << "Input image: " << W << " x " << H << endl;

    // compute padded sizes (next pow2 for both dims)
    int Hp = nextPow2(H);
    int Wp = nextPow2(W);
    cout << "Padded to: " << Wp << " x " << Hp << endl;

    size_t realSize = (size_t)Hp * (size_t)Wp;          // padded real grid
    int freqW = Wp/2 + 1;
    long long freqSize = (long long)Hp * freqW;         // R2C complex size

    // prepare host real buffers (float)
    vector<float> h_img(realSize, 0.0f);
    vector<float> h_kernel(realSize, 0.0f);

    // copy input image into top-left of padded buffer
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            h_img[r * Wp + c] = (float)img.at<unsigned char>(r, c); // keep 0..255 range
        }
    }

    // build box blur kernel (size kernel_size x kernel_size) centered and circularly shifted
    int k = max(1, kernel_size | 1); // ensure odd
    int half = k / 2;
    // Place kernel centered at (0,0) by circular shift: for each kernel sample (kr,kc),
    // place at ((-half + kr + Hp) % Hp, (-half + kc + Wp) % Wp)
    float sum = 0.0f;
    for (int kr = 0; kr < k; ++kr) {
        for (int kc = 0; kc < k; ++kc) {
            int rr = ( (kr - half) + Hp ) % Hp;
            int cc = ( (kc - half) + Wp ) % Wp;
            h_kernel[rr * Wp + cc] = 1.0f;
            sum += 1.0f;
        }
    }
    // normalize kernel to sum = 1.0 (so brightness preserved)
    if (sum != 0.0f) {
        for (size_t i = 0; i < realSize; ++i) h_kernel[i] /= sum;
    }

    // Allocate device real arrays
    float *d_img = nullptr, *d_ker = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_img, sizeof(float) * realSize));
    CUDA_CHECK(cudaMalloc((void**)&d_ker, sizeof(float) * realSize));
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(float) * realSize));

    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), sizeof(float) * realSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ker, h_kernel.data(), sizeof(float) * realSize, cudaMemcpyHostToDevice));

    // allocate complex frequency buffers
    cufftComplex *d_fimg = nullptr, *d_fker = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_fimg, sizeof(cufftComplex) * freqSize));
    CUDA_CHECK(cudaMalloc((void**)&d_fker, sizeof(cufftComplex) * freqSize));

    // create CUFFT plans
    cufftHandle planR2C = 0, planC2R = 0;
    CUFFT_CHECK(cufftPlan2d(&planR2C, Hp, Wp, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&planC2R, Hp, Wp, CUFFT_C2R));

    // forward transforms
    CUFFT_CHECK(cufftExecR2C(planR2C, d_img, d_fimg));
    CUFFT_CHECK(cufftExecR2C(planR2C, d_ker, d_fker));

    // multiply in frequency domain
    int threads = 256;
    long long blocks = (freqSize + threads - 1) / threads;
    pointwiseMul<<< (int)blocks, threads >>>(d_fimg, d_fker, (int)freqSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // inverse transform (result in d_out, real)
    CUFFT_CHECK(cufftExecC2R(planC2R, d_fimg, d_out));
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back full Hp x Wp real result to host
    vector<float> h_out(realSize);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(float) * realSize, cudaMemcpyDeviceToHost));

    // cuFFT inverse is unnormalized: values are scaled by (Hp*Wp). We normalized kernel already to sum=1,
    // so divide by Hp*Wp to get correct pixel magnitudes in same 0..255 range.
    float invScale = 1.0f / (float)(Hp * Wp);

    // crop to original size and convert to uchar
    Mat out(H, W, CV_8UC1);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float v = h_out[r * Wp + c] * invScale; // now should be in 0..255 (approx)
            unsigned char uc = clamp255f(v);
            out.at<unsigned char>(r, c) = uc;
        }
    }

    // Ensure output directory exists and write PNG
    system("mkdir -p output");
    if (!imwrite(outPath, out)) {
        cerr << "Failed to write output PNG: " << outPath << endl;
    } else {
        cout << "Saved output: " << outPath << endl;
    }

    // cleanup
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaFree(d_img);
    cudaFree(d_ker);
    cudaFree(d_out);
    cudaFree(d_fimg);
    cudaFree(d_fker);

    return 0;
}
