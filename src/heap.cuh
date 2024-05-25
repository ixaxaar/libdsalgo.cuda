#ifndef CUDA_HEAP_CUH
#define CUDA_HEAP_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "util.cuh"

enum HeapType
{
    MIN_HEAP,
    MAX_HEAP
};

template <typename T>
class CudaHeap
{
private:
    T *data;
    int size;
    int capacity;
    HeapType heapType;

    __device__ void heapifyUp(int index);
    __device__ void heapifyDown(int index);

public:
    __host__ CudaHeap(int capacity, HeapType heapType);
    __host__ ~CudaHeap();

    __device__ void insert(T value);
    __device__ T extract();
    __device__ T peek() const;
    __device__ bool isEmpty() const;
    __device__ bool isFull() const;

private:
    __host__ void destroy();
};

template <typename T>
__host__ CudaHeap<T>::CudaHeap(int capacity, HeapType heapType)
    : capacity(capacity), heapType(heapType), size(0)
{
    🟩(cudaMalloc(&data, capacity * sizeof(T)));
}

template <typename T>
__host__ CudaHeap<T>::~CudaHeap()
{
    destroy();
}

template <typename T>
__device__ void CudaHeap<T>::insert(T value)
{
    if (isFull())
    {
        printf("Heap overflow\n");
        return;
    }
    data[size] = value;
    heapifyUp(size);
    size++;
}

template <typename T>
__device__ T CudaHeap<T>::extract()
{
    if (isEmpty())
    {
        printf("Heap underflow\n");
        return T();
    }
    T root = data[0];
    data[0] = data[size - 1];
    size--;
    heapifyDown(0);
    return root;
}

template <typename T>
__device__ T CudaHeap<T>::peek() const
{
    if (!isEmpty())
    {
        return data[0];
    }
    else
    {
        printf("Heap is empty\n");
        return T();
    }
}

template <typename T>
__device__ bool CudaHeap<T>::isEmpty() const
{
    return size == 0;
}

template <typename T>
__device__ bool CudaHeap<T>::isFull() const
{
    return size == capacity;
}

template <typename T>
__device__ void CudaHeap<T>::heapifyUp(int index)
{
    int parentIndex = (index - 1) / 2;
    if (index && (heapType == MIN_HEAP ? data[index] < data[parentIndex] : data[index] > data[parentIndex]))
    {
        T temp = data[index];
        data[index] = data[parentIndex];
        data[parentIndex] = temp;
        heapifyUp(parentIndex);
    }
}

template <typename T>
__device__ void CudaHeap<T>::heapifyDown(int index)
{
    int leftChild = 2 * index + 1;
    int rightChild = 2 * index + 2;
    int chosenChild = index;

    if (leftChild < size && (heapType == MIN_HEAP ? data[leftChild] < data[chosenChild] : data[leftChild] > data[chosenChild]))
    {
        chosenChild = leftChild;
    }
    if (rightChild < size && (heapType == MIN_HEAP ? data[rightChild] < data[chosenChild] : data[rightChild] > data[chosenChild]))
    {
        chosenChild = rightChild;
    }
    if (chosenChild != index)
    {
        T temp = data[index];
        data[index] = data[chosenChild];
        data[chosenChild] = temp;
        heapifyDown(chosenChild);
    }
}

template <typename T>
__host__ void CudaHeap<T>::destroy()
{
    cudaFree(data);
}

#endif // CUDA_HEAP_CUH
