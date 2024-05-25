#ifndef CUDA_AVL_TREE_CUH
#define CUDA_AVL_TREE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "util.cuh"

template <typename T>
class CudaAVLTree
{
private:
    TreeNode<T> *root;

public:
    __host__ CudaAVLTree();
    __host__ ~CudaAVLTree();

    __device__ void insert(T value);
    __device__ bool search(T value) const;
    __device__ void inOrderTraversal(void (*func)(const T &)) const;

private:
    __device__ int height(TreeNode<T> *node) const;
    __device__ int getBalance(TreeNode<T> *node) const;
    __device__ TreeNode<T> *insert(TreeNode<T> *node, T value);
    __device__ bool search(TreeNode<T> *node, T value) const;
    __device__ void inOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const;
    __device__ TreeNode<T> *rotateRight(TreeNode<T> *y);
    __device__ TreeNode<T> *rotateLeft(TreeNode<T> *x);
    __host__ void destroy(TreeNode<T> *node);
};

template <typename T>
__host__ CudaAVLTree<T>::CudaAVLTree() : root(nullptr) {}

template <typename T>
__host__ CudaAVLTree<T>::~CudaAVLTree()
{
    destroy(root);
}

template <typename T>
__device__ int CudaAVLTree<T>::height(TreeNode<T> *node) const
{
    if (node == nullptr)
        return 0;
    return node->height;
}

template <typename T>
__device__ int CudaAVLTree<T>::getBalance(TreeNode<T> *node) const
{
    if (node == nullptr)
        return 0;
    return height(node->left) - height(node->right);
}

template <typename T>
__device__ TreeNode<T> *CudaAVLTree<T>::rotateRight(TreeNode<T> *y)
{
    TreeNode<T> *x = y->left;
    TreeNode<T> *T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    return x;
}

template <typename T>
__device__ TreeNode<T> *CudaAVLTree<T>::rotateLeft(TreeNode<T> *x)
{
    TreeNode<T> *y = x->right;
    TreeNode<T> *T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    return y;
}

template <typename T>
__device__ TreeNode<T> *CudaAVLTree<T>::insert(TreeNode<T> *node, T value)
{
    if (node == nullptr)
    {
        TreeNode<T> *newNode;
        CHECK_CUDA_ERROR(cudaMalloc(&newNode, sizeof(TreeNode<T>)));
        newNode->data = value;
        newNode->height = 1;
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
    else
    {
        return node; // Equal values are not allowed in BST
    }

    node->height = 1 + max(height(node->left), height(node->right));

    int balance = getBalance(node);

    // Left Left Case
    if (balance > 1 && value < node->left->data)
    {
        return rotateRight(node);
    }

    // Right Right Case
    if (balance < -1 && value > node->right->data)
    {
        return rotateLeft(node);
    }

    // Left Right Case
    if (balance > 1 && value > node->left->data)
    {
        node->left = rotateLeft(node->left);
        return rotateRight(node);
    }

    // Right Left Case
    if (balance < -1 && value < node->right->data)
    {
        node->right = rotateRight(node->right);
        return rotateLeft(node);
    }

    return node;
}

template <typename T>
__device__ void CudaAVLTree<T>::insert(T value)
{
    root = insert(root, value);
}

template <typename T>
__device__ bool CudaAVLTree<T>::search(T value) const
{
    return search(root, value);
}

template <typename T>
__device__ bool CudaAVLTree<T>::search(TreeNode<T> *node, T value) const
{
    if (node == nullptr)
        return false;

    if (node->data == value)
        return true;

    if (value < node->data)
        return search(node->left, value);
    return search(node->right, value);
}

template <typename T>
__device__ void CudaAVLTree<T>::inOrderTraversal(void (*func)(const T &)) const
{
    inOrderTraversal(root, func);
}

template <typename T>
__device__ void CudaAVLTree<T>::inOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const
{
    if (node != nullptr)
    {
        inOrderTraversal(node->left, func);
        func(node->data);
        inOrderTraversal(node->right, func);
    }
}

template <typename T>
__host__ void CudaAVLTree<T>::destroy(TreeNode<T> *node)
{
    if (node != nullptr)
    {
        destroy(node->left);
        destroy(node->right);
        cudaFree(node);
    }
}

#endif // CUDA_AVL_TREE_CUH
