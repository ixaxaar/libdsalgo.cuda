import scala.collection.mutable
import scala.collection.mutable.PriorityQueue

object DijkstraAlgorithm {

  case class Node(vertex: Int, distance: Int)

  implicit val ord: Ordering[Node] = Ordering.by(-_.distance)

  /**
   * Finds the shortest paths from the source vertex to all other vertices using Dijkstra's algorithm.
   *
   * @param graph  The adjacency matrix representing the graph.
   * @param source The source vertex.
   * @return The array containing the shortest distances from the source to each vertex.
   */
  def dijkstra(graph: Array[Array[Int]], source: Int): Array[Int] = {
    val n = graph.length
    val dist = Array.fill(n)(Int.MaxValue)
    val visited = mutable.Set[Int]()

    // Priority queue to store (vertex, distance) pairs
    val pq = PriorityQueue[Node]()
    dist(source) = 0
    pq.enqueue(Node(source, 0))

    while (pq.nonEmpty) {
      val Node(u, d) = pq.dequeue()

      if (!visited(u)) {
        visited.add(u)

        // Update distances for all adjacent vertices
        for (v <- graph.indices if graph(u)(v) != 0 && !visited(v)) {
          val newDist = dist(u) + graph(u)(v)
          if (newDist < dist(v)) {
            dist(v) = newDist
            pq.enqueue(Node(v, newDist))
          }
        }
      }
    }
    dist
  }

  def main(args: Array[String]): Unit = {
    // Example graph represented as an adjacency matrix
    val graph = Array(
      Array(0, 10, 0, 30, 100),
      Array(10, 0, 50, 0, 0),
      Array(0, 50, 0, 20, 10),
      Array(30, 0, 20, 0, 60),
      Array(100, 0, 10, 60, 0)
    )
    val source = 0
    // Perform Dijkstra's algorithm and print the result
    val dist = dijkstra(graph, source)

    println("Vertex\tDistance from Source")
    for (i <- dist.indices) {
      println(s"$i\t${dist(i)}")
    }
  }
}
