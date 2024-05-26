import scala.collection.mutable.ArrayBuffer
import scala.util.Sorting

object KruskalAlgorithm {

  case class Edge(u: Int, v: Int, weight: Int)

  /**
   * Finds the Minimum Spanning Tree (MST) of a graph using Kruskal's algorithm.
   *
   * @param n     The number of vertices in the graph.
   * @param edges A sequence of edges representing the graph.
   * @return A sequence of edges that form the MST.
   */
  def kruskal(n: Int, edges: Seq[Edge]): Seq[Edge] = {
    val parent = (0 until n).toArray
    val rank = Array.fill(n)(0)

    /**
     * Finds the root of the set containing x using path compression.
     *
     * @param x The element whose set root is to be found.
     * @return The root of the set containing x.
     */
    def find(x: Int): Int = {
      if (parent(x) != x) parent(x) = find(parent(x))
      parent(x)
    }

    /**
     * Unites the sets containing x and y using union by rank.
     *
     * @param x An element in the first set.
     * @param y An element in the second set.
     */
    def union(x: Int, y: Int): Unit = {
      val rootX = find(x)
      val rootY = find(y)
      if (rootX != rootY) {
        if (rank(rootX) > rank(rootY)) {
          parent(rootY) = rootX
        } else if (rank(rootX) < rank(rootY)) {
          parent(rootX) = rootY
        } else {
          parent(rootY) = rootX
          rank(rootX) += 1
        }
      }
    }

    // Sort edges by their weight
    val sortedEdges = edges.sortBy(_.weight)
    val mst = ArrayBuffer[Edge]()

    // Iterate over sorted edges and add them to the MST if they don't form a cycle
    for (Edge(u, v, weight) <- sortedEdges) {
      if (find(u) != find(v)) {
        mst.append(Edge(u, v, weight))
        union(u, v)
      }
    }

    mst
  }

  def main(args: Array[String]): Unit = {
    // Example graph represented as a sequence of edges
    val edges = Seq(
      Edge(0, 1, 10),
      Edge(0, 3, 30),
      Edge(0, 4, 100),
      Edge(1, 2, 50),
      Edge(2, 3, 20),
      Edge(2, 4, 10),
      Edge(3, 4, 60)
    )
    val n = 5
    // Find the MST using Kruskal's algorithm and print the result
    val mst = kruskal(n, edges)

    println("Edges in the Minimum Spanning Tree (MST):")
    mst.foreach { case Edge(u, v, weight) => println(s"($u, $v) -> $weight") }
  }
}
