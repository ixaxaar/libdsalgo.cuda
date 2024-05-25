1. Sorting Algorithms:

   - Bubble Sort:
     - Simple comparison-based sorting algorithm
     - Repeatedly swaps adjacent elements if they are in the wrong order
     - Time complexity: O(n^2), Space complexity: O(1)
     - Inefficient for large datasets, but useful for educational purposes
   - Insertion Sort:
     - Builds the final sorted array one element at a time
     - Inserts each element into its proper position in the sorted portion of the array
     - Time complexity: O(n^2), Space complexity: O(1)
     - Efficient for small datasets or nearly sorted arrays
   - Merge Sort:
     - Divide-and-conquer algorithm that recursively divides the array into halves
     - Merges the sorted halves to produce the final sorted array
     - Time complexity: O(n log n), Space complexity: O(n)
     - Stable sorting algorithm and efficient for large datasets
   - Quick Sort:
     - Divide-and-conquer algorithm that selects a pivot element
     - Partitions the array around the pivot and recursively sorts the sub-arrays
     - Time complexity: O(n log n) on average, O(n^2) in the worst case, Space complexity: O(log n)
     - Efficient for large datasets and in-place sorting

2. Searching Algorithms:

   - Linear Search:
     - Sequentially checks each element of the array until a match is found or the end is reached
     - Time complexity: O(n), Space complexity: O(1)
     - Simple but inefficient for large datasets
   - Binary Search:
     - Divides the sorted array in half and discards the half where the target cannot lie
     - Repeatedly halves the search space until the target is found or the search space is empty
     - Time complexity: O(log n), Space complexity: O(1)
     - Efficient for searching in sorted arrays

3. Graph Traversal Algorithms:

   - Breadth-First Search (BFS):
     - Explores all the neighboring nodes before moving to the next level neighbors
     - Uses a queue data structure to keep track of the nodes to visit
     - Time complexity: O(V + E), where V is the number of vertices and E is the number of edges
     - Useful for finding the shortest path in unweighted graphs and detecting cycles
   - Depth-First Search (DFS):
     - Explores as far as possible along each branch before backtracking
     - Uses a stack data structure (implicit in recursive implementation) to keep track of the nodes to visit
     - Time complexity: O(V + E), where V is the number of vertices and E is the number of edges
     - Useful for detecting cycles, topological sorting, and exploring connected components

4. Dynamic Programming:

   - Technique for solving complex problems by breaking them down into simpler subproblems
   - Stores the results of subproblems to avoid redundant calculations (memoization)
   - Builds the solution incrementally by combining the solutions of subproblems
   - Applicable when the problem has optimal substructure and overlapping subproblems
   - Examples: Fibonacci sequence, longest common subsequence, knapsack problem

5. Greedy Algorithms:
   - Makes the locally optimal choice at each stage with the hope of finding a global optimum
   - Builds the solution incrementally by making the best choice at each step
   - Applicable when the problem has optimal substructure and the greedy choice property
   - Examples: Huffman coding, Dijkstra's shortest path algorithm, Kruskal's minimum spanning tree algorithm
