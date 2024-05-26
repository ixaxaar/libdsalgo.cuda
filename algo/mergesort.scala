object MergeSort {

  /**
   * Sorts an array of integers using the merge sort algorithm.
   *
   * @param arr The array to be sorted.
   * @return A new array that is sorted.
   */
  def mergesort(arr: Array[Int]): Array[Int] = {
    if (arr.length <= 1) arr
    else {
      val mid = arr.length / 2
      val (left, right) = arr.splitAt(mid)
      merge(mergesort(left), mergesort(right))
    }
  }

  /**
   * Merges two sorted arrays into a single sorted array.
   *
   * @param left  The first sorted array.
   * @param right The second sorted array.
   * @return A merged sorted array.
   */
  def merge(left: Array[Int], right: Array[Int]): Array[Int] = {
    (left, right) match {
      case (Array(), _) => right
      case (_, Array()) => left
      case (Array(lHead, lTail @ _*), Array(rHead, rTail @ _*)) =>
        if (lHead < rHead) lHead +: merge(lTail.toArray, right)
        else rHead +: merge(left, rTail.toArray)
    }
  }

  def main(args: Array[String]): Unit = {
    // Example array
    val array = Array(3, 6, 8, 10, 1, 2, 1)
    // Perform merge sort and print the result
    val sortedArray = mergesort(array)
    println(s"Sorted array: ${sortedArray.mkString(", ")}")
  }
}
