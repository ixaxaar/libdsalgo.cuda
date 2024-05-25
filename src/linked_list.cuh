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

    void mergeSort();
    void quickSort();

private:
    static __device__ void insertKernel(Node<T> **head, const T value);
    static __device__ void removeKernel(Node<T> **head, const T value);
    static __device__ void traverseKernel(Node<T> *head, void (*func)(const T &));
    static __device__ void mapKernel(Node<T> *head, T (*func)(T));
    static __device__ void reduceKernel(Node<T> *head, T *result, T (*reducer)(T, T), T initialValue);

    static __device__ void mergeSortKernel(Node<T> **head);
    static Node<T> *mergeSortUtil(Node<T> *head);
    static Node<T> *sortedMerge(Node<T> *a, Node<T> *b);
    static void split(Node<T> *source, Node<T> **frontRef, Node<T> **backRef);

    static __device__ void quickSortKernel(Node<T> **head);
    static Node<T> *quickSortUtil(Node<T> *head);
    static Node<T> *partition(Node<T> *head, Node<T> **newHead, Node<T> **newEnd);
    static Node<T> *getTail(Node<T> *current);
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
    🟩(cudaDeviceSynchronize());
}

template <typename T>
void CudaLinkedList<T>::remove(const T &value)
{
    removeKernel<<<1, 1>>>(&head, value);
    🟩(cudaDeviceSynchronize());
}

template <typename T>
void CudaLinkedList<T>::traverse(void (*func)(const T &)) const
{
    traverseKernel<<<1, 1>>>(head, func);
    🟩(cudaDeviceSynchronize());
}

template <typename T>
void CudaLinkedList<T>::map(T (*func)(T))
{
    mapKernel<<<1, 1>>>(head, func);
    🟩(cudaDeviceSynchronize());
}

template <typename T>
T CudaLinkedList<T>::reduce(T (*reducer)(T, T), T initialValue) const
{
    T *result;
    🟩(cudaMalloc(&result, sizeof(T)));
    🟩(cudaMemcpy(result, &initialValue, sizeof(T), cudaMemcpyHostToDevice));

    reduceKernel<<<1, 1>>>(head, result, reducer, initialValue);
    🟩(cudaDeviceSynchronize());

    T hostResult;
    🟩(cudaMemcpy(&hostResult, result, sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(result);

    return hostResult;
}

template <typename T>
void CudaLinkedList<T>::mergeSort()
{
    mergeSortKernel<<<1, 1>>>(&head);
    🟩(cudaDeviceSynchronize());
}

template <typename T>
void CudaLinkedList<T>::quickSort()
{
    quickSortKernel<<<1, 1>>>(&head);
    🟩(cudaDeviceSynchronize());
}

template <typename T>
__device__ void CudaLinkedList<T>::insertKernel(Node<T> **head, const T value)
{
    Node<T> *newNode;
    🟩(cudaMalloc(&newNode, sizeof(Node<T>)));
    newNode->data = value;
    newNode->next = *head;
    *head = newNode;
}

template <typename T>
__device__ void CudaLinkedList<T>::removeKernel(Node<T> **head, const T value)
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
__device__ void CudaLinkedList<T>::traverseKernel(Node<T> *head, void (*func)(const T &))
{
    Node<T> *current = head;
    while (current != nullptr)
    {
        func(current->data);
        current = current->next;
    }
}

template <typename T>
__device__ void CudaLinkedList<T>::mapKernel(Node<T> *head, T (*func)(T))
{
    Node<T> *current = head;
    while (current != nullptr)
    {
        current->data = func(current->data);
        current = current->next;
    }
}

template <typename T>
__device__ void CudaLinkedList<T>::reduceKernel(Node<T> *head, T *result, T (*reducer)(T, T), T initialValue)
{
    Node<T> *current = head;
    while (current != nullptr)
    {
        atomicAdd(result, reducer(initialValue, current->data));
        current = current->next;
    }
}

template <typename T>
__device__ void CudaLinkedList<T>::mergeSortKernel(Node<T> **head)
{
    *head = mergeSortUtil(*head);
}

template <typename T>
Node<T> *CudaLinkedList<T>::mergeSortUtil(Node<T> *head)
{
    if (head == nullptr || head->next == nullptr)
        return head;

    Node<T> *a;
    Node<T> *b;
    split(head, &a, &b);

    a = mergeSortUtil(a);
    b = mergeSortUtil(b);

    return sortedMerge(a, b);
}

template <typename T>
Node<T> *CudaLinkedList<T>::sortedMerge(Node<T> *a, Node<T> *b)
{
    if (a == nullptr)
        return b;
    if (b == nullptr)
        return a;

    Node<T> *result;
    if (a->data <= b->data)
    {
        result = a;
        result->next = sortedMerge(a->next, b);
    }
    else
    {
        result = b;
        result->next = sortedMerge(a, b->next);
    }

    return result;
}

template <typename T>
void CudaLinkedList<T>::split(Node<T> *source, Node<T> **frontRef, Node<T> **backRef)
{
    Node<T> *fast;
    Node<T> *slow;
    slow = source;
    fast = source->next;

    while (fast != nullptr)
    {
        fast = fast->next;
        if (fast != nullptr)
        {
            slow = slow->next;
            fast = fast->next;
        }
    }

    *frontRef = source;
    *backRef = slow->next;
    slow->next = nullptr;
}

template <typename T>
__device__ void CudaLinkedList<T>::quickSortKernel(Node<T> **head)
{
    *head = quickSortUtil(*head);
}

template <typename T>
Node<T> *CudaLinkedList<T>::quickSortUtil(Node<T> *head)
{
    if (head == nullptr || head->next == nullptr)
        return head;

    Node<T> *newHead = nullptr;
    Node<T> *newEnd = nullptr;

    Node<T> *pivot = partition(head, &newHead, &newEnd);

    if (newHead != pivot)
    {
        Node<T> *temp = newHead;
        while (temp->next != pivot)
        {
            temp = temp->next;
        }
        temp->next = nullptr;

        newHead = quickSortUtil(newHead);

        temp = getTail(newHead);
        temp->next = pivot;
    }

    pivot->next = quickSortUtil(pivot->next);

    return newHead;
}

template <typename T>
Node<T> *CudaLinkedList<T>::partition(Node<T> *head, Node<T> **newHead, Node<T> **newEnd)
{
    Node<T> *pivot = head;
    Node<T> *prev = nullptr;
    Node<T> *curr = head;
    Node<T> *tail = pivot;

    while (curr != nullptr)
    {
        if (curr->data < pivot->data)
        {
            if (*newHead == nullptr)
                *newHead = curr;

            prev = curr;
            curr = curr->next;
        }
        else
        {
            if (prev != nullptr)
                prev->next = curr->next;
            Node<T> *temp = curr->next;
            curr->next = nullptr;
            tail->next = curr;
            tail = curr;
            curr = temp;
        }
    }

    if (*newHead == nullptr)
        *newHead = pivot;

    *newEnd = tail;

    return pivot;
}

template <typename T>
Node<T> *CudaLinkedList<T>::getTail(Node<T> *current)
{
    while (current != nullptr && current->next != nullptr)
    {
        current = current->next;
    }
    return current;
}

#endif // CUDA_LINKED_LIST_CUH
