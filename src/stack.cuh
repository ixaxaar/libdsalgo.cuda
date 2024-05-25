#ifndef CUDA_STACK_CUH
#define CUDA_STACK_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "linked_list.cuh"
#include "util.cuh"

template <typename T>
class CudaStack
{
private:
    Node<T> *top;

public:
    __host__ CudaStack();
    __host__ ~CudaStack();

    __device__ void push(T value);
    __device__ T pop();
    __device__ T peek() const;
    __device__ bool isEmpty() const;
};

template <typename T>
__host__ CudaStack<T>::CudaStack() : top(nullptr) {}

template <typename T>
__host__ CudaStack<T>::~CudaStack()
{
    Node<T> *current = top;
    while (current != nullptr)
    {
        Node<T> *next = current->next;
        cudaFree(current);
        current = next;
    }
}

template <typename T>
__device__ void CudaStack<T>::push(T value)
{
    Node<T> *newNode;
    🟩(cudaMalloc(&newNode, sizeof(Node<T>)));
    newNode->data = value;
    newNode->next = top;
    top = newNode;
}

template <typename T>
__device__ T CudaStack<T>::pop()
{
    if (!isEmpty())
    {
        Node<T> *temp = top;
        T value = top->data;
        top = top->next;
        cudaFree(temp);
        return value;
    }
    else
    {
        printf("Stack underflow\n");
        return T();
    }
}

template <typename T>
__device__ T CudaStack<T>::peek() const
{
    if (!isEmpty())
    {
        return top->data;
    }
    else
    {
        printf("Stack is empty\n");
        return T();
    }
}

template <typename T>
__device__ bool CudaStack<T>::isEmpty() const
{
    return top == nullptr;
}

#endif // CUDA_STACK_CUH
