import scala.collection.mutable

object DFS {

  /**
   * Performs a depth-first search (DFS) on the given graph starting from the specified start node.
   *
   * @param graph A map representing the adjacency list of the graph. Each key is a node, and the corresponding value
   *              is a list of nodes that are adjacent to the key node.
   * @param start The starting node for the DFS.
   * @return A list of nodes in the order they were visited during the DFS.
   */
  def dfs(graph: Map[Int, List[Int]], start: Int): List[Int] = {
    // Set to keep track of visited nodes
    val visited = mutable.Set[Int]()
    // Stack for the DFS
    val stack = mutable.Stack[Int]()
    // List to store the order of visited nodes
    val result = mutable.ListBuffer[Int]()

    // Initialize the DFS with the start node
    stack.push(start)

    // Continue the DFS until there are no more nodes to explore
    while (stack.nonEmpty) {
      val node = stack.pop()

      if (!visited(node)) {
        visited.add(node)
        result += node

        // Push all unvisited neighbors onto the stack
        for (neighbor <- graph(node)) {
          if (!visited(neighbor)) {
            stack.push(neighbor)
          }
        }
      }
    }

    result.toList
  }

  def main(args: Array[String]): Unit = {
    // Example graph represented as an adjacency list
    val graph = Map(
      0 -> List(1, 2),
      1 -> List(0, 3, 4),
      2 -> List(0, 4),
      3 -> List(1, 5),
      4 -> List(1, 2),
      5 -> List(3)
    )
    val startNode = 0
    // Perform DFS and print the result
    val traversal = dfs(graph, startNode)
    println(s"DFS Traversal starting from node $startNode: ${traversal.mkString(", ")}")
  }
}
