#include "filter_impl.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb {
    uint8_t r, g, b;
};

struct rgbState {
    uint8_t r, g, b, time;
};

__device__ float color_distance(const rgb* p1, const rgbState* p2) {
    return sqrtf((p1->r - p2->r) * (p1->r - p2->r) +
                 (p1->g - p2->g) * (p1->g - p2->g) +
                 (p1->b - p2->b) * (p1->b - p2->b));
}

__global__ void motion_first_frame(rgb* dbuffer_frame, size_t pitch_dbuffer_frame, rgbState* dbuffer_background, size_t pitch_dbuffer_background, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;
    rgb* idxFrame = ((rgb*)((char *)dbuffer_frame + y * pitch_dbuffer_frame));
    rgbState* idxBackground = (rgbState*)((char *)dbuffer_background + y * pitch_dbuffer_background);
    idxBackground[x].r = idxFrame[x].r;
    idxBackground[x].g = idxFrame[x].g;
    idxBackground[x].b = idxFrame[x].b;
    idxBackground[x].time = 0;
}

__global__ void motion_detect(rgb* dbuffer_frame, size_t pitch_dbuffer_frame, rgbState* dbuffer_background, size_t pitch_dbuffer_background, uint8_t* dbuffer_grayscale, size_t pitch_dbuffer_grayscale, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* idxFrame = ((rgb*)((char *)dbuffer_frame + y * pitch_dbuffer_frame));
    rgbState* idxBackground = (rgbState*)((char *)dbuffer_background + y * pitch_dbuffer_background);
    float distance = color_distance(&idxFrame[x], &idxBackground[x]);

    if (distance <= 25.0f)
    {
        idxBackground[x].time = 0;
        dbuffer_grayscale[x + y * pitch_dbuffer_grayscale] = (uint8_t)distance;
    }
    else
    {
        dbuffer_grayscale[x + y * pitch_dbuffer_grayscale] = (uint8_t)distance;
        idxBackground[x].time++;
        if (idxBackground[x].time > 5)
        {
            idxBackground[x].r = idxFrame[x].r;
            idxBackground[x].g = idxFrame[x].g;
            idxBackground[x].b = idxFrame[x].b;
            idxBackground[x].time = 0;
            dbuffer_grayscale[x + y * pitch_dbuffer_grayscale] = (uint8_t)distance;
        }
    }
}

__global__ void erosion_row_major(uint8_t* dsrc_grayscale, size_t dsrc_grayscale_pitch, uint8_t* dgrayscale_column_major, size_t dgrayscale_column_major_pitch, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width-1 || y >= height-1 || x <= 0 || y <= 0)
        return;

    uint8_t eroded = min(min(dsrc_grayscale[y * dsrc_grayscale_pitch + x-1], 
                             dsrc_grayscale[y * dsrc_grayscale_pitch + x]), 
                         dsrc_grayscale[y * dsrc_grayscale_pitch + x+1]);
    dgrayscale_column_major[x * dgrayscale_column_major_pitch + y] = eroded;
}


__global__ void erosion_column_major(uint8_t* dsrc_grayscale_column_major, size_t dsrc_grayscale_column_major_pitch, uint8_t* dgrayscale_row_major, size_t dgrayscale_row_major_pitch, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width-1 || y >= height-1 || x <= 0 || y <= 0)
        return;

    uint8_t eroded = min(min(dsrc_grayscale_column_major[x * dsrc_grayscale_column_major_pitch + y-1], 
                             dsrc_grayscale_column_major[x * dsrc_grayscale_column_major_pitch + y]), 
                         dsrc_grayscale_column_major[x * dsrc_grayscale_column_major_pitch + y+1]);
    dgrayscale_row_major[y * dgrayscale_row_major_pitch + x] = eroded;
}

__global__ void dilation_row_major(uint8_t* dsrc_grayscale, size_t dsrc_grayscale_pitch, uint8_t* dgrayscale_column_major, size_t dgrayscale_column_major_pitch, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width-1 || y >= height-1 || x <= 0 || y <= 0)
        return;

    uint8_t dilated = max(max(dsrc_grayscale[y * dsrc_grayscale_pitch + x-1],
                              dsrc_grayscale[y * dsrc_grayscale_pitch + x]), 
                          dsrc_grayscale[y * dsrc_grayscale_pitch + x+1]);
    dgrayscale_column_major[x * dgrayscale_column_major_pitch + y] = dilated;
}

__global__ void dilation_column_major(uint8_t* dsrc_grayscale_column_major, size_t dsrc_grayscale_column_major_pitch, uint8_t* dgrayscale_row_major, size_t dgrayscale_row_major_pitch, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width-1 || y >= height-1 || x <= 0 || y <= 0)
        return;

    uint8_t dilated = max(max(dsrc_grayscale_column_major[x * dsrc_grayscale_column_major_pitch + y-1], 
                              dsrc_grayscale_column_major[x * dsrc_grayscale_column_major_pitch + y]), 
                          dsrc_grayscale_column_major[x * dsrc_grayscale_column_major_pitch + y+1]);
    dgrayscale_row_major[y * dgrayscale_row_major_pitch + x] = dilated;
}

__global__ void hysterisis(uint8_t* dbuffer_row_major, size_t pitch_dbuffer_row_major, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width-1 || y >= height-1 || x <= 0 || y <= 0)
        return;
    
    if (dbuffer_row_major[x + y * pitch_dbuffer_row_major] >= 30)
        dbuffer_row_major[x + y * pitch_dbuffer_row_major] = 255;
    else if (dbuffer_row_major[x + y * pitch_dbuffer_row_major] < 4)
        dbuffer_row_major[x + y * pitch_dbuffer_row_major] = 0;
    else
    {
        int i = -1;
        int j = -1;
        while (j <= 1)
        {
            while (i >= 1)
            {
                if (dbuffer_row_major[(x + i) + (y + j) * pitch_dbuffer_row_major] >= 30)
                {
                    dbuffer_row_major[x + y * pitch_dbuffer_row_major] = 255;
                    return;
                }
                i++;
            }
            j++;
        }
        dbuffer_row_major[x + y * pitch_dbuffer_row_major] = 0;
    }
}

__global__ void apply_red(rgb* dbuffer_frame, size_t pitch_dbuffer_frame, uint8_t* dbuffer_row_major, size_t pitch_dbuffer_row_major, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;
    

    rgb* idxFrame = ((rgb*)((char *)dbuffer_frame + y * pitch_dbuffer_frame));
    if (dbuffer_row_major[x + y * pitch_dbuffer_row_major] > 0)
        idxFrame[x].r = min(255, (idxFrame[x].r + (uint8_t)(0.5f * 255)));
}

extern "C" {
static uint8_t* dBuffer = nullptr;
static uint8_t* dbuffer_greyscale_row_major = nullptr;
static uint8_t* dbuffer_greyscale_column_major = nullptr;
static rgbState* dbuffer_background = nullptr;

static size_t pitch = 0;
static size_t pitch_dbuffer_greyscale_row_major = 0;
static size_t pitch_dbuffer_greyscale_column_major = 0;
static size_t pitch_dbuffer_background = 0;
static int current_width = 0;
static int current_height = 0;

static cudaStream_t stream = nullptr;

void filter_init() {
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err);
}

void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride) {
    assert(sizeof(rgb) == pixel_stride);
    cudaError_t err;

    if (!stream) {
        filter_init();
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                (height + blockSize.y - 1) / blockSize.y);

    if (!dBuffer || width != current_width || height != current_height) {
        if (dBuffer) {
            err = cudaFree(dBuffer);
            CHECK_CUDA_ERROR(err);
        }
        if (dbuffer_greyscale_row_major) {
            err = cudaFree(dbuffer_greyscale_row_major);
            CHECK_CUDA_ERROR(err);
        }
        if (dbuffer_greyscale_column_major) {
            err = cudaFree(dbuffer_greyscale_column_major);
            CHECK_CUDA_ERROR(err);
        }
        if (dbuffer_background) {
            err = cudaFree(dbuffer_background);
            CHECK_CUDA_ERROR(err);
        }

        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMallocPitch(&dbuffer_greyscale_row_major, &pitch_dbuffer_greyscale_row_major, 
                             width * sizeof(uint8_t), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMallocPitch(&dbuffer_greyscale_column_major, &pitch_dbuffer_greyscale_column_major,
                             height * sizeof(uint8_t), width);
        CHECK_CUDA_ERROR(err);

        err = cudaMallocPitch(&dbuffer_background, &pitch_dbuffer_background, width * sizeof(rgbState), height);
        CHECK_CUDA_ERROR(err);
       
        err = cudaMemcpy2DAsync(dBuffer, pitch, src_buffer, src_stride, 
                                width * sizeof(rgb), height, cudaMemcpyHostToDevice, stream);
        CHECK_CUDA_ERROR(err);
        
        motion_first_frame<<<gridSize, blockSize, 0, stream>>>((rgb*)dBuffer, pitch, dbuffer_background, pitch_dbuffer_background, width, height);

        current_width = width;
        current_height = height;
    }
    else
{

        err = cudaMemcpy2DAsync(dBuffer, pitch, src_buffer, src_stride, 
                                width * sizeof(rgb), height, cudaMemcpyHostToDevice, stream);
        CHECK_CUDA_ERROR(err);

        motion_detect<<<gridSize, blockSize, 0, stream>>>((rgb*)dBuffer, pitch, dbuffer_background, pitch_dbuffer_background, dbuffer_greyscale_row_major, pitch_dbuffer_greyscale_row_major, width, height);

        erosion_row_major<<<(256, 1), blockSize, 0, stream>>>(
            dbuffer_greyscale_row_major, pitch_dbuffer_greyscale_row_major,
            dbuffer_greyscale_column_major,
            pitch_dbuffer_greyscale_column_major, width, height);

        erosion_column_major<<<(256, 1), blockSize, 0, stream>>>(
            dbuffer_greyscale_column_major, pitch_dbuffer_greyscale_column_major,
            dbuffer_greyscale_row_major, pitch_dbuffer_greyscale_row_major,
            width, height);

        dilation_row_major<<<(256, 1), blockSize, 0, stream>>>(
            dbuffer_greyscale_row_major, pitch_dbuffer_greyscale_row_major,
            dbuffer_greyscale_column_major, pitch_dbuffer_greyscale_column_major,
            width, height);

        dilation_column_major<<<(256, 1), blockSize, 0, stream>>>(
            dbuffer_greyscale_column_major, pitch_dbuffer_greyscale_column_major,
            dbuffer_greyscale_row_major, pitch_dbuffer_greyscale_row_major,
            width, height);

        hysterisis<<<gridSize, blockSize, 0, stream>>>(dbuffer_greyscale_row_major, pitch_dbuffer_greyscale_row_major, width, height);

        apply_red<<<gridSize, blockSize, 0, stream>>>((rgb*)dBuffer, pitch, dbuffer_greyscale_row_major, pitch_dbuffer_greyscale_row_major, width, height);
    }

    err = cudaMemcpy2DAsync(src_buffer, src_stride, dBuffer, pitch, 
                            width * sizeof(rgb), height, cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA_ERROR(err);

    err = cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR(err);
}

void filter_cleanup() {
    if (dBuffer) cudaFree(dBuffer);
    if (dbuffer_greyscale_row_major) cudaFree(dbuffer_greyscale_row_major);
    if (dbuffer_greyscale_column_major) cudaFree(dbuffer_greyscale_column_major);
    
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    dBuffer = nullptr;
    dbuffer_greyscale_row_major = nullptr;
    dbuffer_greyscale_column_major = nullptr;
    current_width = 0;
    current_height = 0;
}
}
