# CUDA Data Structures and Algorithms

## TLDR

I was trying to revise data structures and algorithms for an interview so I thought why not do it in cuda and take help from LLMs to do everything in one evening.

## Motivation

Preparing for technical interviews often involves revising and practicing data structures and algorithms. To make this process more challenging and educational, we chose to implement these concepts using CUDA, allowing us to explore parallel computing and GPU acceleration. This project demonstrates how traditional data structures and algorithms can be efficiently implemented in CUDA, offering both a revision of core concepts and an introduction to CUDA programming.

## Implemented Data Structures and Algorithms

### 1. Arrays

- **CudaArray Class**: Supports basic operations such as filling, mapping, and reducing. Additionally, includes merge sort and quick sort functionalities. [src/array.cuh](src/array.cuh)

### 2. Linked Lists

- **CudaLinkedList Class**: Provides methods for insertion, removal, traversal, mapping, and reduction. Also implements merge sort and quick sort. [src/linked_list.cuh](src/linked_list.cuh)

### 3. Stacks

- **CudaStack Class**: A dynamic stack implemented using linked lists. [src/stack.cuh](src/stack.cuh)

### 4. Queues

- **CudaQueue Class**: A dynamic queue implemented using linked lists. [src/queue.cuh](src/queue.cuh)

### 5. Trees

- **CudaBinaryTree Class**: Basic binary tree with insertion, searching, and traversal. [src/binary_tree.cuh](src/binary_tree.cuh)
- **CudaBST Class**: Binary Search Tree with insertion, searching, and traversal. [src/bst.cuh](src/bst.cuh)
- **CudaAVLTree Class**: Self-balancing AVL tree with insertion, searching, and balancing operations. [src/avl.cuh](src/avl.cuh)
- **CudaRBTree Class**: Red-Black Tree with insertion, searching, and balancing operations. [src/rbtree.cuh](src/rbtree.cuh)

### 6. Graphs

- **CudaGraph Class**: Represents a graph using an adjacency list. Supports BFS and DFS traversals. [src/graph.cuh](src/graph.cuh)

### 7. Hash Tables

- **CudaHashTable Class**: Implements a hash table with separate chaining for collision handling. [src/hashmap.cuh](src/hashmap.cuh)

### 8. Heaps

- **CudaHeap Class**: A binary heap (min-heap or max-heap) with insertion, extraction, and heapify operations. [src/heap.cuh](src/heap.cuh)

## How to Run

### Compilation

To compile any of the CUDA programs, use the `nvcc` compiler with the following command format:

```sh
nvcc <source_file>.cu -o <output_file> -std=c++11
```

### Execution

Run the compiled executable:

```sh
./<output_file>
```

## Example Usage

Here is an example of how to use the `CudaArray` class for merge sort and quick sort:

### `main_sort.cu`

```cpp
#include "cuda_array.cuh"

void printArray(int* data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

int main() {
    int data1[] = {38, 27, 43, 3, 9, 82, 10};
    int size1 = sizeof(data1) / sizeof(data1[0]);

    CudaArray<int> arr1(size1);
    cudaMemcpy(arr1.getData(), data1, size1 * sizeof(int), cudaMemcpyHostToDevice);

    printf("Original array for Merge Sort:\n");
    printArray(data1, size1);

    arr1.mergeSort();

    cudaMemcpy(data1, arr1.getData(), size1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sorted array using Merge Sort:\n");
    printArray(data1, size1);

    int data2[] = {38, 27, 43, 3, 9, 82, 10};
    int size2 = sizeof(data2) / sizeof(data2[0]);

    CudaArray<int> arr2(size2);
    cudaMemcpy(arr2.getData(), data2, size2 * sizeof(int), cudaMemcpyHostToDevice);

    printf("Original array for Quick Sort:\n");
    printArray(data2, size2);

    arr2.quickSort();

    cudaMemcpy(data2, arr2.getData(), size2 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sorted array using Quick Sort:\n");
    printArray(data2, size2);

    return 0;
}
```

### Compilation

```sh
nvcc main_sort.cu -o sort -std=c++11
```

### Execution

```sh
./sort
```

## Conclusion

This project demonstrates the power of combining fundamental data structures and algorithms with CUDA programming. By leveraging the parallel processing capabilities of CUDA, we can achieve efficient implementations of these core concepts. The assistance of a Language Model (LLM) enabled rapid development and completion of the project within an evening, showcasing the potential of AI-driven programming assistance.

Happy coding and best of luck with your interview preparations!
