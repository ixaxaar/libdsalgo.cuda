#ifndef CUDA_QUEUE_CUH
#define CUDA_QUEUE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "linked_list.cuh"
#include "util.cuh"

template <typename T>
class CudaQueue
{
private:
    Node<T> *front;
    Node<T> *rear;

public:
    __host__ CudaQueue();
    __host__ ~CudaQueue();

    __device__ void enqueue(T value);
    __device__ T dequeue();
    __device__ T peek() const;
    __device__ bool isEmpty() const;
};

template <typename T>
__host__ CudaQueue<T>::CudaQueue() : front(nullptr), rear(nullptr) {}

template <typename T>
__host__ CudaQueue<T>::~CudaQueue()
{
    Node<T> *current = front;
    while (current != nullptr)
    {
        Node<T> *next = current->next;
        cudaFree(current);
        current = next;
    }
}

template <typename T>
__device__ void CudaQueue<T>::enqueue(T value)
{
    Node<T> *newNode;
    CHECK_CUDA_ERROR(cudaMalloc(&newNode, sizeof(Node<T>)));
    newNode->data = value;
    newNode->next = nullptr;
    if (isEmpty())
    {
        front = rear = newNode;
    }
    else
    {
        rear->next = newNode;
        rear = newNode;
    }
}

template <typename T>
__device__ T CudaQueue<T>::dequeue()
{
    if (!isEmpty())
    {
        Node<T> *temp = front;
        T value = front->data;
        front = front->next;
        if (front == nullptr)
        {
            rear = nullptr;
        }
        cudaFree(temp);
        return value;
    }
    else
    {
        printf("Queue underflow\n");
        return T();
    }
}

template <typename T>
__device__ T CudaQueue<T>::peek() const
{
    if (!isEmpty())
    {
        return front->data;
    }
    else
    {
        printf("Queue is empty\n");
        return T();
    }
}

template <typename T>
__device__ bool CudaQueue<T>::isEmpty() const
{
    return front == nullptr;
}

#endif // CUDA_QUEUE_CUH
