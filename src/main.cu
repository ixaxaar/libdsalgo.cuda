#include "array.cuh"
#include "avl.cuh"
#include "bst.cuh"
#include "btree.cuh"
#include "graph.cuh"
#include "hashmap.cuh"
#include "heap.cuh"
#include "linked_list.cuh"
#include "queue.cuh"
#include "rbtree.cuh"
#include "stack.cuh"
#include "util.cuh"

void printArray(int *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", data[i]);
    }
    printf("\n");
}

__device__ void printData(const int &data)
{
    printf("%d ", data);
}
