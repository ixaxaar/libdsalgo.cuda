#ifndef CUDA_HASH_TABLE_CUH
#define CUDA_HASH_TABLE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <list>
#include "util.cuh"

template <typename K, typename V>
struct HashNode
{
    K key;
    V value;
    HashNode *next;
};

template <typename K, typename V>
class CudaHashTable
{
private:
    HashNode<K, V> **table;
    int capacity;

    __device__ int hashFunction(K key) const;

public:
    __host__ CudaHashTable(int capacity);
    __host__ ~CudaHashTable();

    __device__ void insert(K key, V value);
    __device__ bool search(K key, V &value) const;
    __device__ void remove(K key);

private:
    __host__ void destroy();
};

template <typename K, typename V>
__device__ int CudaHashTable<K, V>::hashFunction(K key) const
{
    return key % capacity;
}

template <typename K, typename V>
__host__ CudaHashTable<K, V>::CudaHashTable(int capacity) : capacity(capacity)
{
    🟩(cudaMalloc(&table, capacity * sizeof(HashNode<K, V> *)));
    🟩(cudaMemset(table, 0, capacity * sizeof(HashNode<K, V> *)));
}

template <typename K, typename V>
__host__ CudaHashTable<K, V>::~CudaHashTable()
{
    destroy();
}

template <typename K, typename V>
__device__ void CudaHashTable<K, V>::insert(K key, V value)
{
    int hashIndex = hashFunction(key);
    HashNode<K, V> *newNode;
    🟩(cudaMalloc(&newNode, sizeof(HashNode<K, V>)));
    newNode->key = key;
    newNode->value = value;
    newNode->next = nullptr;

    if (table[hashIndex] == nullptr)
    {
        table[hashIndex] = newNode;
    }
    else
    {
        HashNode<K, V> *current = table[hashIndex];
        while (current->next != nullptr)
        {
            current = current->next;
        }
        current->next = newNode;
    }
}

template <typename K, typename V>
__device__ bool CudaHashTable<K, V>::search(K key, V &value) const
{
    int hashIndex = hashFunction(key);
    HashNode<K, V> *current = table[hashIndex];
    while (current != nullptr)
    {
        if (current->key == key)
        {
            value = current->value;
            return true;
        }
        current = current->next;
    }
    return false;
}

template <typename K, typename V>
__device__ void CudaHashTable<K, V>::remove(K key)
{
    int hashIndex = hashFunction(key);
    HashNode<K, V> *current = table[hashIndex];
    HashNode<K, V> *prev = nullptr;

    while (current != nullptr && current->key != key)
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
        table[hashIndex] = current->next;
    }

    cudaFree(current);
}

template <typename K, typename V>
__host__ void CudaHashTable<K, V>::destroy()
{
    for (int i = 0; i < capacity; ++i)
    {
        HashNode<K, V> *current = table[i];
        while (current != nullptr)
        {
            HashNode<K, V> *next = current->next;
            cudaFree(current);
            current = next;
        }
    }
    cudaFree(table);
}

#endif // CUDA_HASH_TABLE_CUH
