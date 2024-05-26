import scala.collection.mutable
import scala.collection.mutable.PriorityQueue

object PrimAlgorithm {

  case class Edge(node: Int, weight: Int)

  /**
   * Finds the Minimum Spanning Tree (MST) of a graph using Prim's algorithm.
   *
   * @param graph The adjacency list representing the graph.
   * @param start The starting node for Prim's algorithm.
   * @return A sequence of edges that form the MST.
   */
  def prims(graph: Map[Int, List[Edge]], start: Int): List[Edge] = {
    implicit val ord: Ordering[Edge] = Ordering.by(-_.weight)
    val pq = PriorityQueue[Edge]()
    val visited = mutable.Set[Int]()
    val mst = mutable.ListBuffer[Edge]()

    /**
     * Adds all edges from the given node to the priority queue.
     *
     * @param node The node from which edges are added.
     */
    def addEdges(node: Int): Unit = {
      visited.add(node)
      for (edge <- graph(node) if !visited(edge.node)) {
        pq.enqueue(edge)
      }
    }

    addEdges(start)

    while (pq.nonEmpty) {
      val Edge(node, weight) = pq.dequeue()
      if (!visited(node)) {
        mst.append(Edge(node, weight))
        addEdges(node)
      }
    }

    mst.toList
  }

  def main(args: Array[String]): Unit = {
    // Example graph represented as an adjacency list
    val graph = Map(
      0 -> List(Edge(1, 10), Edge(3, 30), Edge(4, 100)),
      1 -> List(Edge(0, 10), Edge(2, 50)),
      2 -> List(Edge(1, 50), Edge(3, 20), Edge(4, 10)),
      3 -> List(Edge(0, 30), Edge(2, 20), Edge(4, 60)),
      4 -> List(Edge(
