#ifndef CUDA_LINKED_LIST_CUH
#define CUDA_LINKED_LIST_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "util.cuh"

template <typename T>
struct Node
{
    T data;
    Node *next;
};

template <typename T>
class CudaLinkedList
{
private:
    Node<T> *head;

public:
    CudaLinkedList();
    ~CudaLinkedList();

    void insert(const T &value);
    void remove(const T &value);
    void traverse(void (*func)(const T &)) const;
    void map(T (*func)(T));
    T reduce(T (*reducer)(T, T), T initialValue) const;

private:
    static __global__ void insertKernel(Node<T> **head, const T value);
    static __global__ void removeKernel(Node<T> **head, const T value);
    static __global__ void traverseKernel(Node<T> *head, void (*func)(const T &));
    static __global__ void mapKernel(Node<T> *head, T (*func)(T));
    static __global__ void reduceKernel(Node<T> *head, T *result, T (*reducer)(T, T), T initialValue);
};

template <typename T>
CudaLinkedList<T>::CudaLinkedList() : head(nullptr) {}

template <typename T>
CudaLinkedList<T>::~CudaLinkedList()
{
    Node<T> *current = head;
    while (current != nullptr)
    {
        Node<T> *next = current->next;
        cudaFree(current);
        current = next;
    }
}

template <typename T>
void CudaLinkedList<T>::insert(const T &value)
{
    insertKernel<<<1, 1>>>(&head, value);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename T>
void CudaLinkedList<T>::remove(const T &value)
{
    removeKernel<<<1, 1>>>(&head, value);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename T>
void CudaLinkedList<T>::traverse(void (*func)(const T &)) const
{
    traverseKernel<<<1, 1>>>(head, func);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename T>
void CudaLinkedList<T>::map(T (*func)(T))
{
    mapKernel<<<1, 1>>>(head, func);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename T>
T CudaLinkedList<T>::reduce(T (*reducer)(T, T), T initialValue) const
{
    T *result;
    CHECK_CUDA_ERROR(cudaMalloc(&result, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemcpy(result, &initialValue, sizeof(T), cudaMemcpyHostToDevice));

    reduceKernel<<<1, 1>>>(head, result, reducer, initialValue);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    T hostResult;
    CHECK_CUDA_ERROR(cudaMemcpy(&hostResult, result, sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(result);

    return hostResult;
}

template <typename T>
__global__ void CudaLinkedList<T>::insertKernel(Node<T> **head, const T value)
{
    Node<T> *newNode;
    CHECK_CUDA_ERROR(cudaMalloc(&newNode, sizeof(Node<T>)));
    newNode->data = value;
    newNode->next = *head;
    *head = newNode;
}

template <typename T>
__global__ void CudaLinkedList<T>::removeKernel(Node<T> **head, const T value)
{
    Node<T> *current = *head;
    Node<T> *prev = nullptr;

    while (current != nullptr && current->data != value)
    {
        prev = current;
        current = current->next;
    }

    if (current == nullptr)
        return;

    if (prev != nullptr)
    {
        prev->next = current->next;
    }
    else
    {
        *head = current->next;
    }

    cudaFree(current);
}

template <typename T>
__global__ void CudaLinkedList<T>::traverseKernel(Node<T> *head, void (*func)(const T &))
{
    Node<T> *current = head;
    while (current != nullptr)
    {
        func(current->data);
        current = current->next;
    }
}

template <typename T>
__global__ void CudaLinkedList<T>::mapKernel(Node<T> *head, T (*func)(T))
{
    Node<T> *current = head;
    while (current != nullptr)
    {
        current->data = func(current->data);
        current = current->next;
    }
}

template <typename T>
__global__ void CudaLinkedList<T>::reduceKernel(Node<T> *head, T *result, T (*reducer)(T, T), T initialValue)
{
    Node<T> *current = head;
    while (current != nullptr)
    {
        atomicAdd(result, reducer(initialValue, current->data));
        current = current->next;
    }
}

#endif // CUDA_LINKED_LIST_CUH
