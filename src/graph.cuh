#ifndef CUDA_GRAPH_CUH
#define CUDA_GRAPH_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include "util.cuh"

template <typename T>
struct GraphNode
{
    T data;
    std::vector<GraphNode *> neighbors;
};

template <typename T>
class CudaGraph
{
private:
    std::vector<GraphNode<T> *> vertices;

public:
    __host__ CudaGraph();
    __host__ ~CudaGraph();

    __host__ void addVertex(T value);
    __host__ void addEdge(T from, T to);

    __global__ void bfs(T startValue, void (*visit)(const T &));
    __global__ void dfs(T startValue, void (*visit)(const T &));

private:
    __device__ void bfsUtil(GraphNode<T> *startNode, void (*visit)(const T &));
    __device__ void dfsUtil(GraphNode<T> *node, void (*visit)(const T &), bool *visited);

    __host__ GraphNode<T> *findNode(T value);
    __host__ void destroy();
};

template <typename T>
__host__ CudaGraph<T>::CudaGraph() {}

template <typename T>
__host__ CudaGraph<T>::~CudaGraph()
{
    destroy();
}

template <typename T>
__host__ void CudaGraph<T>::addVertex(T value)
{
    GraphNode<T> *newNode = new GraphNode<T>{value};
    vertices.push_back(newNode);
}

template <typename T>
__host__ void CudaGraph<T>::addEdge(T from, T to)
{
    GraphNode<T> *fromNode = findNode(from);
    GraphNode<T> *toNode = findNode(to);

    if (fromNode && toNode)
    {
        fromNode->neighbors.push_back(toNode);
    }
    else
    {
        printf("Invalid edge from %d to %d\n", from, to);
    }
}

template <typename T>
__host__ GraphNode<T> *CudaGraph<T>::findNode(T value)
{
    for (auto vertex : vertices)
    {
        if (vertex->data == value)
        {
            return vertex;
        }
    }
    return nullptr;
}

template <typename T>
__device__ void CudaGraph<T>::bfsUtil(GraphNode<T> *startNode, void (*visit)(const T &))
{
    if (startNode == nullptr)
        return;

    std::queue<GraphNode<T> *> q;
    q.push(startNode);

    std::vector<bool> visited(vertices.size(), false);

    while (!q.empty())
    {
        GraphNode<T> *current = q.front();
        q.pop();

        if (!visited[current->data])
        {
            visit(current->data);
            visited[current->data] = true;

            for (auto neighbor : current->neighbors)
            {
                if (!visited[neighbor->data])
                {
                    q.push(neighbor);
                }
            }
        }
    }
}

template <typename T>
__global__ void CudaGraph<T>::bfs(T startValue, void (*visit)(const T &))
{
    GraphNode<T> *startNode = nullptr;
    for (auto vertex : vertices)
    {
        if (vertex->data == startValue)
        {
            startNode = vertex;
            break;
        }
    }
    bfsUtil(startNode, visit);
}

template <typename T>
__device__ void CudaGraph<T>::dfsUtil(GraphNode<T> *node, void (*visit)(const T &), bool *visited)
{
    if (node == nullptr || visited[node->data])
        return;

    visit(node->data);
    visited[node->data] = true;

    for (auto neighbor : node->neighbors)
    {
        if (!visited[neighbor->data])
        {
            dfsUtil(neighbor, visit, visited);
        }
    }
}

template <typename T>
__global__ void CudaGraph<T>::dfs(T startValue, void (*visit)(const T &))
{
    bool *visited;
    CHECK_CUDA_ERROR(cudaMalloc(&visited, vertices.size() * sizeof(bool)));
    cudaMemset(visited, 0, vertices.size() * sizeof(bool));

    GraphNode<T> *startNode = nullptr;
    for (auto vertex : vertices)
    {
        if (vertex->data == startValue)
        {
            startNode = vertex;
            break;
        }
    }
    dfsUtil(startNode, visit, visited);
    CHECK_CUDA_ERROR(cudaFree(visited));
}

template <typename T>
__host__ void CudaGraph<T>::destroy()
{
    for (auto vertex : vertices)
    {
        delete vertex;
    }
    vertices.clear();
}

#endif // CUDA_GRAPH_CUH
