# CUDA Data Structures and Algorithms

## TLDR

I was trying to revise data structures and algorithms for an interview so I thought why not do it in cuda and take help from LLMs to do everything in one evening.

## Motivation

Because giving entrance exams at this age is boring as shit.

## Contains

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
make
```

Happy coding and best of luck with your interview preparations and have fun learning CUDA!
