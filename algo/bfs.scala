import scala.collection.mutable

object BFS {

  /**
   * Performs a breadth-first search (BFS) on the given graph starting from the specified start node.
   *
   * @param graph A map representing the adjacency list of the graph. Each key is a node, and the corresponding value
   *              is a list of nodes that are adjacent to the key node.
   * @param start The starting node for the BFS.
   * @return A list of nodes in the order they were visited during the BFS.
   */
  def bfs(graph: Map[Int, List[Int]], start: Int): List[Int] = {
    // Set to keep track of visited nodes
    val visited = mutable.Set[Int]()
    // Queue for the BFS
    val queue = mutable.Queue[Int]()
    // List to store the order of visited nodes
    val result = mutable.ListBuffer[Int]()

    // Initialize the BFS with the start node
    queue.enqueue(start)
    visited.add(start)

    // Continue the BFS until there are no more nodes to explore
    while (queue.nonEmpty) {
      val node = queue.dequeue()
      result += node

      // Enqueue all unvisited neighbors and mark them as visited
      for (neighbor <- graph(node) if !visited(neighbor)) {
        queue.enqueue(neighbor)
        visited.add(neighbor)
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
    // Perform BFS and print the result
    val traversal = bfs(graph, startNode)
    println(s"BFS Traversal starting from node $startNode: ${traversal.mkString(", ")}")
  }
}
