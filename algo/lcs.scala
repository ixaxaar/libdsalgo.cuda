object LongestCommonSubsequence {

  /**
   * Finds the longest common subsequence (LCS) of two given strings using dynamic programming.
   *
   * @param x The first string.
   * @param y The second string.
   * @return The longest common subsequence of the two strings.
   */
  def lcs(x: String, y: String): String = {
    val m = x.length
    val n = y.length
    val dp = Array.ofDim[Int](m + 1, n + 1)

    // Build the dp table
    for (i <- 1 to m; j <- 1 to n) {
      if (x(i - 1) == y(j - 1)) dp(i)(j) = dp(i - 1)(j - 1) + 1
      else dp(i)(j) = Math.max(dp(i - 1)(j), dp(i)(j - 1))
    }

    // Construct the LCS from the dp table
    val sb = new StringBuilder
    var i = m
    var j = n
    while (i > 0 && j > 0) {
      if (x(i - 1) == y(j - 1)) {
        sb.append(x(i - 1))
        i -= 1
        j -= 1
      } else if (dp(i - 1)(j) >= dp(i)(j - 1)) {
        i -= 1
      } else {
        j -= 1
      }
    }

    sb.reverse.toString
  }

  def main(args: Array[String]): Unit = {
    // Example strings
    val x = "AGGTAB"
    val y = "GXTXAYB"
    // Find the LCS of the two strings and print the result
    val result = lcs(x, y)
    println(s"Longest Common Subsequence of '$x' and '$y' is: '$result'")
  }
}
