#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>

// =======================================================
// プラグインデータ構造
// =======================================================
typedef struct {
    VSNode* node;
    VSVideoInfo vi;
    int tff;   // 1 = Top field first, 0 = Bottom field first
    int mode;  // 0 = double-rate (bob), 1 = single-rate
} CudaDeintData;

// =======================================================
// CUDA カーネル: マルチタップ補間
// =======================================================
__global__ void deintKernel(const uint8_t* src, uint8_t* dst,
    int w, int h, size_t stride,
    int useTop, int isChroma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    bool missing = (useTop ? (y % 2 == 1) : (y % 2 == 0));

    if (!missing) {
        dst[y * stride + x] = src[y * stride + x];
    }
    else {
        if (!isChroma) {
            // 輝度: 4タップ (近傍重視)
            int sum = 0, weight = 0;
            for (int dy = -3; dy <= 3; dy += 2) {
                int yy = y + dy;
                if (yy < 0) yy = 0;
                if (yy >= h) yy = h - 1;
                int wgt = (abs(dy) == 1) ? 4 : 1;
                sum += src[yy * stride + x] * wgt;
                weight += wgt;
            }
            dst[y * stride + x] = (sum + weight / 2) / weight;
        }
        else {
            // 色成分: 2タップ
            int y0 = (y > 0) ? y - 1 : 0;
            int y1 = (y + 1 < h) ? y + 1 : h - 1;
            int v0 = src[y0 * stride + x];
            int v1 = src[y1 * stride + x];
            dst[y * stride + x] = (v0 + v1 + 1) / 2;
        }
    }
}

// =======================================================
// ホスト側ラッパー
// =======================================================
static void runDeintKernel(const uint8_t* sp, uint8_t* dp,
    int w, int h, int stride,
    int useTop, int isChroma)
{
    size_t frame_size = (size_t)stride * h;

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;

    cudaMalloc(&d_src, frame_size);
    cudaMalloc(&d_dst, frame_size);
    cudaMemcpy(d_src, sp, frame_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    deintKernel << <blocks, threads >> > (d_src, d_dst, w, h, stride, useTop, isChroma);
    cudaDeviceSynchronize();

    cudaMemcpy(dp, d_dst, frame_size, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

// =======================================================
// フレーム処理
// =======================================================
static const VSFrame* VS_CC cudaDeintGetFrame(
    int n, int activationReason, void* instanceData, void**,
    VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    CudaDeintData* d = (CudaDeintData*)instanceData;

    if (activationReason == arInitial) {
        int srcN = (d->mode == 0) ? n / 2 : n;
        vsapi->requestFrameFilter(srcN, d->node, frameCtx);
        return nullptr;
    }

    if (activationReason == arAllFramesReady) {
        int srcN = (d->mode == 0) ? n / 2 : n;
        const VSFrame* src = vsapi->getFrameFilter(srcN, d->node, frameCtx);

        VSFrame* dst = vsapi->newVideoFrame(&d->vi.format,
            d->vi.width, d->vi.height,
            src, core);

        bool useTop = (d->mode == 0)
            ? ((n % 2 == 0) == (d->tff == 1))
            : (d->tff == 1);

        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            int w = vsapi->getFrameWidth(src, plane);
            int h = vsapi->getFrameHeight(src, plane);
            int stride = vsapi->getStride(src, plane);

            const uint8_t* sp = vsapi->getReadPtr(src, plane);
            uint8_t* dp = vsapi->getWritePtr(dst, plane);

            runDeintKernel(sp, dp, w, h, stride, useTop, (plane > 0));
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

// =======================================================
// Free
// =======================================================
static void VS_CC cudaDeintFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    CudaDeintData* d = (CudaDeintData*)instanceData;
    if (d->node) vsapi->freeNode(d->node);
    free(d);
}

// =======================================================
// Create
// =======================================================
static void VS_CC cudaDeintCreate(const VSMap* in, VSMap* out, void* userData,
    VSCore* core, const VSAPI* vsapi)
{
    int err;
    CudaDeintData* d = (CudaDeintData*)malloc(sizeof(CudaDeintData));

    d->node = vsapi->mapGetNode(in, "clip", 0, &err);
    if (err) {
        vsapi->mapSetError(out, "CudaDeinterlacer: clip is required.");
        free(d);
        return;
    }

    d->vi = *vsapi->getVideoInfo(d->node);

    d->tff = (int)vsapi->mapGetInt(in, "tff", 0, &err);
    if (err) d->tff = 1;

    d->mode = (int)vsapi->mapGetInt(in, "mode", 0, &err);
    if (err) d->mode = 0;

    if (d->mode == 0) {
        d->vi.fpsNum *= 2;
        d->vi.numFrames *= 2;
    }

    VSFilterDependency deps[] = { { d->node, rpGeneral } };
    vsapi->createVideoFilter(out, "CudaDeinterlacer", &d->vi,
        cudaDeintGetFrame, cudaDeintFree,
        fmParallel, deps, 1, d, core);
}

// =======================================================
// Init
// =======================================================
VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.example.cudadeint", "cdeint",
        "CUDA High-Quality Deinterlacer (bob/single)",
        VS_MAKE_VERSION(1, 0),
        VAPOURSYNTH_API_VERSION,
        0, plugin);

    vspapi->registerFunction("CudaDeinterlacer",
        "clip:vnode;mode:int:opt;tff:int:opt;",
        "clip:vnode;",
        cudaDeintCreate, NULL, plugin);
}
