# CUDA Data Structures and Algorithms

## TLDR

I was trying to revise data structures and algorithms for an interview so I thought why not do it in cuda and take help from LLMs to do everything in one evening.

## Motivation

Because giving entrance exams at this age is meaningless and boring as shit.

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

### Time and Space Complexities

| Data Structure / Algorithm | Operation                 | Time Complexity | Space Complexity |
|----------------------------|---------------------------|------------------|------------------|
| **Array**                  | Access                    | O(1)             | O(n)             |
|                            | Search                    | O(n)             | O(n)             |
|                            | Insertion (at end)        | O(1)             | O(n)             |
|                            | Deletion (at end)         | O(1)             | O(n)             |
| **Linked List**            | Access                    | O(n)             | O(n)             |
|                            | Search                    | O(n)             | O(n)             |
|                            | Insertion (at head)       | O(1)             | O(n)             |
|                            | Deletion (at head)        | O(1)             | O(n)             |
| **Stack (Array-based)**    | Push                      | O(1)             | O(n)             |
|                            | Pop                       | O(1)             | O(n)             |
|                            | Peek                      | O(1)             | O(n)             |
| **Stack (Linked List-based)** | Push                   | O(1)             | O(n)             |
|                            | Pop                       | O(1)             | O(n)             |
|                            | Peek                      | O(1)             | O(n)             |
| **Queue (Array-based)**    | Enqueue                   | O(1)             | O(n)             |
|                            | Dequeue                   | O(1)             | O(n)             |
| **Queue (Linked List-based)** | Enqueue                | O(1)             | O(n)             |
|                            | Dequeue                   | O(1)             | O(n)             |
| **Binary Tree**            | Search                    | O(n)             | O(n)             |
|                            | Insertion                 | O(n)             | O(n)             |
|                            | Deletion                  | O(n)             | O(n)             |
| **Binary Search Tree**     | Search                    | O(log n)         | O(n)             |
|                            | Insertion                 | O(log n)         | O(n)             |
|                            | Deletion                  | O(log n)         | O(n)             |
| **AVL Tree**               | Search                    | O(log n)         | O(n)             |
|                            | Insertion                 | O(log n)         | O(n)             |
|                            | Deletion                  | O(log n)         | O(n)             |
| **Red-Black Tree**         | Search                    | O(log n)         | O(n)             |
|                            | Insertion                 | O(log n)         | O(n)             |
|                            | Deletion                  | O(log n)         | O(n)             |
| **Graph (Adjacency List)** | Add Vertex                | O(1)             | O(V + E)         |
|                            | Add Edge                  | O(1)             | O(V + E)         |
|                            | Remove Vertex             | O(V + E)         | O(V + E)         |
|                            | Remove Edge               | O(E)             | O(V + E)         |
|                            | BFS                       | O(V + E)         | O(V)             |
|                            | DFS                       | O(V + E)         | O(V)             |
| **Hash Table**             | Insert                    | O(1) (average)   | O(n)             |
|                            | Delete                    | O(1) (average)   | O(n)             |
|                            | Search                    | O(1) (average)   | O(n)             |
| **Heap (Min/Max Heap)**    | Insert                    | O(log n)         | O(n)             |
|                            | Delete                    | O(log n)         | O(n)             |
|                            | Peek                      | O(1)             | O(n)             |
| **Merge Sort**             | Sort                      | O(n log n)       | O(n)             |
| **Quick Sort**             | Sort                      | O(n log n) (avg) | O(log n)         |
| **Kruskal's Algorithm**    | Minimum Spanning Tree (MST)   | O(E log E)             | O(E + V)               |
| **Prim's Algorithm**       | Minimum Spanning Tree (MST)   | O(E logV + V log V)| O(V)                  |
| **Dijkstra's Algorithm**   | Shortest Path                 | O(V^2) or O(E + V log V)| O(V)                  |
| **Bellman-Ford Algorithm** | Shortest Path                 | O(VE)                  | O(V)                   |

### Notes:
- **V**: Number of vertices in the graph.
- **E**: Number of edges in the graph.
- For **Quick Sort**, the average case time complexity is O(n log n), but the worst-case complexity is O(n^2).
