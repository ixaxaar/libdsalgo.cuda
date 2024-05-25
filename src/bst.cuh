#ifndef CUDA_BST_CUH
#define CUDA_BST_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "util.cuh"

template <typename T>
class CudaBST
{
private:
    TreeNode<T> *root;

public:
    __host__ CudaBST();
    __host__ ~CudaBST();

    __device__ void insert(T value);
    __device__ bool search(T value) const;
    __device__ void inOrderTraversal(void (*func)(const T &)) const;
    __device__ void preOrderTraversal(void (*func)(const T &)) const;
    __device__ void postOrderTraversal(void (*func)(const T &)) const;

private:
    __device__ TreeNode<T> *insert(TreeNode<T> *node, T value);
    __device__ bool search(TreeNode<T> *node, T value) const;
    __device__ void inOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const;
    __device__ void preOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const;
    __device__ void postOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const;
    __host__ void destroy(TreeNode<T> *node);
};

template <typename T>
__host__ CudaBST<T>::CudaBST() : root(nullptr) {}

template <typename T>
__host__ CudaBST<T>::~CudaBST()
{
    destroy(root);
}

template <typename T>
__device__ void CudaBST<T>::insert(T value)
{
    root = insert(root, value);
}

template <typename T>
__device__ bool CudaBST<T>::search(T value) const
{
    return search(root, value);
}

template <typename T>
__device__ void CudaBST<T>::inOrderTraversal(void (*func)(const T &)) const
{
    inOrderTraversal(root, func);
}

template <typename T>
__device__ void CudaBST<T>::preOrderTraversal(void (*func)(const T &)) const
{
    preOrderTraversal(root, func);
}

template <typename T>
__device__ void CudaBST<T>::postOrderTraversal(void (*func)(const T &)) const
{
    postOrderTraversal(root, func);
}

template <typename T>
__device__ TreeNode<T> *CudaBST<T>::insert(TreeNode<T> *node, T value)
{
    if (node == nullptr)
    {
        TreeNode<T> *newNode;
        CHECK_CUDA_ERROR(cudaMalloc(&newNode, sizeof(TreeNode<T>)));
        newNode->data = value;
        newNode->left = nullptr;
        newNode->right = nullptr;
        return newNode;
    }

    if (value < node->data)
    {
        node->left = insert(node->left, value);
    }
    else if (value > node->data)
    {
        node->right = insert(node->right, value);
    }

    return node;
}

template <typename T>
__device__ bool CudaBST<T>::search(TreeNode<T> *node, T value) const
{
    if (node == nullptr)
    {
        return false;
    }

    if (node->data == value)
    {
        return true;
    }

    if (value < node->data)
    {
        return search(node->left, value);
    }
    else
    {
        return search(node->right, value);
    }
}

template <typename T>
__device__ void CudaBST<T>::inOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const
{
    if (node != nullptr)
    {
        inOrderTraversal(node->left, func);
        func(node->data);
        inOrderTraversal(node->right, func);
    }
}

template <typename T>
__device__ void CudaBST<T>::preOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const
{
    if (node != nullptr)
    {
        func(node->data);
        preOrderTraversal(node->left, func);
        preOrderTraversal(node->right, func);
    }
}

template <typename T>
__device__ void CudaBST<T>::postOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const
{
    if (node != nullptr)
    {
        postOrderTraversal(node->left, func);
        postOrderTraversal(node->right, func);
        func(node->data);
    }
}

template <typename T>
__host__ void CudaBST<T>::destroy(TreeNode<T> *node)
{
    if (node != nullptr)
    {
        destroy(node->left);
        destroy(node->right);
        cudaFree(node);
    }
}

#endif // CUDA_BST_CUH
