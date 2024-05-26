object QuickSort {

  /**
   * Sorts an array of integers using the quicksort algorithm.
   *
   * @param arr The array to be sorted.
   * @return A new array that is sorted.
   */
  def quicksort(arr: Array[Int]): Array[Int] = {
    if (arr.length <= 1) arr
    else {
      val pivot = arr(arr.length / 2)
      Array.concat(
        quicksort(arr.filter(_ < pivot)),
        arr.filter(_ == pivot),
        quicksort(arr.filter(_ > pivot))
      )
    }
  }

  def main(args: Array[String]): Unit = {
    // Example array
    val array = Array(3, 6, 8, 10, 1, 2, 1)
    // Perform quicksort and print the result
    val sortedArray = quicksort(array)
    println(s"Sorted array: ${sortedArray.mkString(", ")}")
  }
}
