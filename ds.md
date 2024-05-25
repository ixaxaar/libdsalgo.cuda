1. Arrays and Strings:

   - Arrays are contiguous blocks of memory that store elements of the same type
   - Access elements by index in O(1) time complexity
   - Search, insertion, and deletion have O(n) time complexity (unless at the end of the array)
   - Dynamic arrays (e.g., vector in C++, ArrayList in Java) automatically resize when needed
   - Strings are essentially character arrays and can be manipulated using various techniques
   - Substring extraction, concatenation, and palindrome checking are common string operations
   - Two pointers or sliding window techniques are useful for solving problems efficiently
   - In-place modification of arrays can save space

2. Linked Lists:

   - Linked lists consist of nodes, each containing a value and a reference to the next node
   - Singly linked lists have a unidirectional link, while doubly linked lists have bidirectional links
   - Access elements by traversing from the head node, which takes O(n) time complexity
   - Insertion and deletion at a given position have O(1) time complexity at the head and O(n) otherwise
   - The runner technique (slow and fast pointers) is useful for cycle detection and finding the middle element
   - Sentinel nodes (dummy nodes) simplify edge case handling by avoiding null checks
   - Reversing a linked list can be done iteratively or recursively

3. Stacks and Queues:

   - Stacks follow the LIFO (Last-In-First-Out) principle
   - Common operations: push (insert), pop (remove), and peek (get top element) in O(1) time complexity
   - Can be implemented using an array or a linked list
   - Useful for function call Stack, expression evaluation, and backtracking problems
   - Queues follow the FIFO (First-In-First-Out) principle
   - Common operations: enqueue (insert), dequeue (remove), and peek (get front element) in O(1) time complexity
   - Can be implemented using an array or a linked list
   - Useful for breadth-first search (BFS) and cache implementations

4. Trees and Graphs:

   - Trees are hierarchical data structures with a root node and child nodes
   - Binary trees have at most two children per node (left and right)
   - Binary search trees (BSTs) have the property that the left subtree values are less than the node, and the right subtree values are greater
   - Balanced trees (e.g., AVL trees, Red-Black trees) ensure O(log n) time complexity for search, insertion, and deletion
   - Traversal methods: Breadth-First Search (BFS) and Depth-First Search (DFS) - Inorder, Preorder, Postorder
   - Graphs are a collection of nodes (vertices) and edges connecting them
   - Can be represented using an adjacency matrix or adjacency list
   - BFS and DFS are used for graph traversal and solving various problems
   - Dijkstra's algorithm for shortest path and Kruskal's or Prim's algorithm for minimum spanning tree

5. Hash Tables:

   - Hash tables provide fast O(1) average time complexity for insertion, deletion, and search
   - Elements are stored based on their hash code, which is computed using a hash function
   - Collisions can be resolved using techniques like chaining (linked lists) or open addressing (probing)
   - Load factor (ratio of occupied slots to total slots) affects performance and resizing
   - Useful for implementing caches, symbol tables, and dictionaries

6. Heaps:
   - Heaps are complete binary trees that satisfy the heap property
   - Max heap: parent nodes are greater than or equal to child nodes
   - Min heap: parent nodes are less than or equal to child nodes
   - Insertion and deletion take O(log n) time complexity
   - The top element (root) can be accessed in O(1) time complexity
   - Heapify operation is used to maintain the heap property after insertion or deletion
   - Useful for implementing priority queues and sorting algorithms like Heap Sort
