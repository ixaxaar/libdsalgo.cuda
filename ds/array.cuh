#ifndef CUDA_ARRAY_CUH
#define CUDA_ARRAY_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "util.cuh"

template <typename T>
class CudaArray
{
private:
    T *data;
    size_t size;

public:
    CudaArray(size_t size);
    ~CudaArray();

    T *getData() const;
    size_t getSize() const;

    void fill(const T value);
    void map(T (*func)(T));
    T reduce(T (*reducer)(T, T), T initialValue);

    void mergeSort();
    void quickSort();

private:
    static __global__ void fillKernel(T *data, T value, size_t size);
    static __global__ void mapKernel(T *data, T (*func)(T), size_t size);
    static __global__ void reduceKernel(T *data, T *result, size_t size, T (*reducer)(T, T), T initialValue);

    static __global__ void mergeSortKernel(T *data, T *temp, int left, int right);
    static __global__ void mergeKernel(T *data, T *temp, int left, int mid, int right);

    static __global__ void quickSortKernel(T *data, int low, int high);
    static __device__ int partition(T *data, int low, int high);
};

template <typename T>
CudaArray<T>::CudaArray(size_t size) : size(size)
{
    cudaMalloc(&data, size * sizeof(T));
}

template <typename T>
CudaArray<T>::~CudaArray()
{
    cudaFree(data);
}

template <typename T>
T *CudaArray<T>::getData() const
{
    return data;
}

template <typename T>
size_t CudaArray<T>::getSize() const
{
    return size;
}

template <typename T>
void CudaArray<T>::fill(const T value)
{
    fillKernel<<<(size + 255) / 256, 256>>>(data, value, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaArray<T>::map(T (*func)(T))
{
    mapKernel<<<(size + 255) / 256, 256>>>(data, func, size);
    cudaDeviceSynchronize();
}

template <typename T>
T CudaArray<T>::reduce(T (*reducer)(T, T), T initialValue)
{
    T *result;
    cudaMalloc(&result, sizeof(T));
    cudaMemcpy(result, &initialValue, sizeof(T), cudaMemcpyHostToDevice);

    reduceKernel<<<(size + 255) / 256, 256, 256 * sizeof(T)>>>(data, result, size, reducer, initialValue);
    cudaDeviceSynchronize();

    T hostResult;
    cudaMemcpy(&hostResult, result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(result);

    return hostResult;
}

template <typename T>
void CudaArray<T>::mergeSort()
{
    T *d_temp;
    cudaMalloc(&d_temp, size * sizeof(T));

    mergeSortKernel<<<1, 1>>>(data, d_temp, 0, size - 1);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

template <typename T>
void CudaArray<T>::quickSort()
{
    quickSortKernel<<<1, 1>>>(data, 0, size - 1);
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void CudaArray<T>::fillKernel(T *data, T value, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = value;
    }
}

template <typename T>
__global__ void CudaArray<T>::mapKernel(T *data, T (*func)(T), size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = func(data[idx]);
    }
}

template <typename T>
__global__ void CudaArray<T>::reduceKernel(T *data, T *result, size_t size, T (*reducer)(T, T), T initialValue)
{
    extern __shared__ T sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = (idx < size) ? data[idx] : initialValue;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sharedData[tid] = reducer(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(result, sharedData[0]);
    }
}

template <typename T>
__global__ void CudaArray<T>::mergeKernel(T *data, T *temp, int left, int mid, int right)
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right)
    {
        if (data[i] <= data[j])
        {
            temp[k++] = data[i++];
        }
        else
        {
            temp[k++] = data[j++];
        }
    }

    while (i <= mid)
    {
        temp[k++] = data[i++];
    }

    while (j <= right)
    {
        temp[k++] = data[j++];
    }

    for (i = left; i <= right; i++)
    {
        data[i] = temp[i];
    }
}

template <typename T>
__global__ void CudaArray<T>::mergeSortKernel(T *data, T *temp, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        mergeSortKernel<<<1, 1>>>(data, temp, left, mid);
        cudaDeviceSynchronize();
        mergeSortKernel<<<1, 1>>>(data, temp, mid + 1, right);
        cudaDeviceSynchronize();
        mergeKernel<<<1, 1>>>(data, temp, left, mid, right);
        cudaDeviceSynchronize();
    }
}

template <typename T>
__device__ int CudaArray<T>::partition(T *data, int low, int high)
{
    T pivot = data[high];
    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        if (data[j] <= pivot)
        {
            i++;
            T temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    T temp = data[i + 1];
    data[i + 1] = data[high];
    data[high] = temp;
    return i + 1;
}

template <typename T>
__global__ void CudaArray<T>::quickSortKernel(T *data, int low, int high)
{
    if (low < high)
    {
        int pi = partition(data, low, high);

        quickSortKernel<<<1, 1>>>(data, low, pi - 1);
        cudaDeviceSynchronize();
        quickSortKernel<<<1, 1>>>(data, pi + 1, high);
        cudaDeviceSynchronize();
    }
}

#endif // CUDA_ARRAY_CUH
