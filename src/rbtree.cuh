#ifndef CUDA_RB_TREE_CUH
#define CUDA_RB_TREE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "util.cuh"

enum Color
{
    RED,
    BLACK
};

template <typename T>
class CudaRBTree
{
private:
    TreeNode<T> *root;

public:
    __host__ CudaRBTree();
    __host__ ~CudaRBTree();

    __device__ void insert(T value);
    __device__ bool search(T value) const;
    __device__ void inOrderTraversal(void (*func)(const T &)) const;

private:
    __device__ TreeNode<T> *insert(TreeNode<T> *node, TreeNode<T> *parent, T value);
    __device__ void fixInsert(TreeNode<T> *node);
    __device__ void rotateLeft(TreeNode<T> *&root, TreeNode<T> *&node);
    __device__ void rotateRight(TreeNode<T> *&root, TreeNode<T> *&node);
    __device__ bool search(TreeNode<T> *node, T value) const;
    __device__ void inOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const;
    __host__ void destroy(TreeNode<T> *node);
};

template <typename T>
__host__ CudaRBTree<T>::CudaRBTree() : root(nullptr) {}

template <typename T>
__host__ CudaRBTree<T>::~CudaRBTree()
{
    destroy(root);
}

template <typename T>
__device__ TreeNode<T> *CudaRBTree<T>::insert(TreeNode<T> *node, TreeNode<T> *parent, T value)
{
    if (node == nullptr)
    {
        TreeNode<T> *newNode;
        CHECK_CUDA_ERROR(cudaMalloc(&newNode, sizeof(TreeNode<T>)));
        newNode->data = value;
        newNode->color = RED;
        newNode->left = nullptr;
        newNode->right = nullptr;
        newNode->parent = parent;
        return newNode;
    }

    if (value < node->data)
    {
        node->left = insert(node->left, node, value);
    }
    else if (value > node->data)
    {
        node->right = insert(node->right, node, value);
    }

    return node;
}

template <typename T>
__device__ void CudaRBTree<T>::insert(T value)
{
    root = insert(root, nullptr, value);
    fixInsert(root);
}

template <typename T>
__device__ void CudaRBTree<T>::fixInsert(TreeNode<T> *node)
{
    TreeNode<T> *parent = nullptr;
    TreeNode<T> *grandparent = nullptr;

    while ((node != root) && (node->color != BLACK) && (node->parent->color == RED))
    {
        parent = node->parent;
        grandparent = node->parent->parent;

        if (parent == grandparent->left)
        {
            TreeNode<T> *uncle = grandparent->right;

            if (uncle != nullptr && uncle->color == RED)
            {
                grandparent->color = RED;
                parent->color = BLACK;
                uncle->color = BLACK;
                node = grandparent;
            }
            else
            {
                if (node == parent->right)
                {
                    rotateLeft(root, parent);
                    node = parent;
                    parent = node->parent;
                }
                rotateRight(root, grandparent);
                std::swap(parent->color, grandparent->color);
                node = parent;
            }
        }
        else
        {
            TreeNode<T> *uncle = grandparent->left;

            if (uncle != nullptr && uncle->color == RED)
            {
                grandparent->color = RED;
                parent->color = BLACK;
                uncle->color = BLACK;
                node = grandparent;
            }
            else
            {
                if (node == parent->left)
                {
                    rotateRight(root, parent);
                    node = parent;
                    parent = node->parent;
                }
                rotateLeft(root, grandparent);
                std::swap(parent->color, grandparent->color);
                node = parent;
            }
        }
    }
    root->color = BLACK;
}

template <typename T>
__device__ void CudaRBTree<T>::rotateLeft(TreeNode<T> *&root, TreeNode<T> *&node)
{
    TreeNode<T> *nodeRight = node->right;

    node->right = nodeRight->left;

    if (node->right != nullptr)
    {
        node->right->parent = node;
    }

    nodeRight->parent = node->parent;

    if (node->parent == nullptr)
    {
        root = nodeRight;
    }
    else if (node == node->parent->left)
    {
        node->parent->left = nodeRight;
    }
    else
    {
        node->parent->right = nodeRight;
    }

    nodeRight->left = node;
    node->parent = nodeRight;
}

template <typename T>
__device__ void CudaRBTree<T>::rotateRight(TreeNode<T> *&root, TreeNode<T> *&node)
{
    TreeNode<T> *nodeLeft = node->left;

    node->left = nodeLeft->right;

    if (node->left != nullptr)
    {
        node->left->parent = node;
    }

    nodeLeft->parent = node->parent;

    if (node->parent == nullptr)
    {
        root = nodeLeft;
    }
    else if (node == node->parent->left)
    {
        node->parent->left = nodeLeft;
    }
    else
    {
        node->parent->right = nodeLeft;
    }

    nodeLeft->right = node;
    node->parent = nodeLeft;
}

template <typename T>
__device__ bool CudaRBTree<T>::search(T value) const
{
    return search(root, value);
}

template <typename T>
__device__ bool CudaRBTree<T>::search(TreeNode<T> *node, T value) const
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
__device__ void CudaRBTree<T>::inOrderTraversal(void (*func)(const T &)) const
{
    inOrderTraversal(root, func);
}

template <typename T>
__device__ void CudaRBTree<T>::inOrderTraversal(TreeNode<T> *node, void (*func)(const T &)) const
{
    if (node != nullptr)
    {
        inOrderTraversal(node->left, func);
        func(node->data);
        inOrderTraversal(node->right, func);
    }
}

template <typename T>
__host__ void CudaRBTree<T>::destroy(TreeNode<T> *node)
{
    if (node != nullptr)
    {
        destroy(node->left);
        destroy(node->right);
        cudaFree(node);
    }
}

#endif // CUDA_RB_TREE_CUH
