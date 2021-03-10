# -----------------------------------------------------------------------
"""
Max Rectangle in Binary Matrix

Given a 2D binary matrix filled with 0’s and 1’s, find the largest rectangle containing all ones and return its area.

Bonus if you can solve it in O(n^2) or less.

Example :

A : [  1 1 1
       0 1 1
       1 0 0
    ]

Output : 4

As the max area rectangle is created by the 2x2 rectangle created by (0,1), (0,2), (1,1) and (1,2)

"""


class Solution:
    # @param A : list of list of integers
    # @return an integer
    def maximalRectangle(self, A):

        curr_row = [0] * len(A[0])
        max_area = 0
        for row in range(len(A)):
            for col in range(len(A[0])):
                if A[row][col] == 0:
                    curr_row[col] = 0
                else:
                    curr_row[col] += A[row][col]

            max_area = max(max_area, calc_histo_area(curr_row))

        return max_area

    # time O(n * m)
    # space O(m)


def calc_histo_area(arr):
    stack = []
    curr_area = 0
    max_area = 0

    for i in range(len(arr)):
        curr_height = arr[i]
        while stack and arr[stack[-1]] > curr_height:
            curr_area = calc_area(arr, stack, i)
            max_area = max(max_area, curr_area)
        stack.append(i)
    i += 1
    while stack:
        curr_area = calc_area(arr, stack, i)
        max_area = max(max_area, curr_area)

    return max_area


def calc_area(arr, stack, i):
    top = stack.pop()
    if not stack:
        return arr[top] * i
    else:
        return arr[top] * (i - stack[-1] - 1)


# -----------------------------------------------------------------------
"""
Distinct Subsequences

Given two sequences A, B, count number of unique ways in sequence A, to form a subsequence that is identical to the sequence B.

Subsequence : A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, “ACE” is a subsequence of “ABCDE” while “AEC” is not).

Input Format:

The first argument of input contains a string, A.
The second argument of input contains a string, B.
Output Format:

Return an integer representing the answer as described in the problem statement.
Constraints:

1 <= length(A), length(B) <= 700
Example :

Input 1:
    A = "abc"
    B = "abc"
    
Output 1:
    1

Explanation 1:
    Both the strings are equal.

Input 2:
    A = "rabbbit" 
    B = "rabbit"

Output 2:
    3

Explanation 2:
    These are the possible removals of characters:
        => A = "ra_bbit" 
        => A = "rab_bit" 
        => A = "rabb_it"
        
    Note: "_" marks the removed character.
"""


# top down approach
class Solution:
    # @param A : string
    # @param B : string
    # @return an integer
    def numDistinct(self, A, B):
        if len(B) > len(A):
            return 0

        memo = [[None for _ in range(len(B))] for _ in range(len(A))]

        return recur_count_distinct(A, B, 0, 0, memo)

    # time O(n * m)
    # space O(n * m)


def recur_count_distinct(s, t, idx1, idx2, memo):
    if idx2 == len(t):
        return 1

    if idx1 == len(s):
        return 0

    if memo[idx1][idx2]:
        return memo[idx1][idx2]

    if s[idx1] != t[idx2]:
        memo[idx1][idx2] = recur_count_distinct(s, t, idx1 + 1, idx2, memo)
    else:
        memo[idx1][idx2] = recur_count_distinct(s, t, idx1 + 1, idx2 + 1, memo) + recur_count_distinct(s, t, idx1 + 1,
                                                                                                       idx2, memo)
    return memo[idx1][idx2]


# bottom up approach
class Solution:
    # @param A : string
    # @param B : string
    # @return an integer
    def numDistinct(self, A, B):

        if len(B) > len(A):
            return 0

        counts = [[0 for _ in range(len(A) + 1)] for _ in range(len(B) + 1)]

        for col in range(len(A) + 1):
            counts[0][col] = 1

        for row in range(1, len(B) + 1):
            counts[row][0] = 0

        for row in range(1, len(B) + 1):
            for col in range(1, len(A) + 1):
                if A[col - 1] == B[row - 1]:
                    counts[row][col] = counts[row][col - 1] + counts[row - 1][col - 1]
                else:
                    counts[row][col] = counts[row][col - 1]

        return counts[len(B)][len(A)]

    # time O(n * m)
    # space O(n * m)


# -----------------------------------------------------------------------
"""
Palindrome Partitioning II

Given a string A, partition A such that every substring of the partition is a palindrome.

Return the minimum cuts needed for a palindrome partitioning of A.



Input Format:

The first and the only argument contains the string A.
Output Format:

Return an integer, representing the answer as described in the problem statement.
Constraints:

1 <= length(A) <= 501
Examples:

Input 1:
    A = "aba"

Output 1:
    0

Explanation 1:
    "aba" is already a palindrome, so no cuts are needed.

Input 2:
    A = "aab"
    
Output 2:
    1

Explanation 2:
    Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
"""


class Solution:
    # @param A : string
    # @return an integer
    def minCut(self, A):
        n = len(A)
        if is_palindrome(A, 0, n - 1):
            return 0

        memo = [[None for _ in range(n)] for _ in range(n)]

        return recur_min_cut(A, 0, n - 1, memo)

    # time O(n^3)
    # space O(n^2)


def recur_min_cut(string, i, j, memo):
    if i == j:
        return 0

    if memo[i][j] != None:
        return memo[i][j]

    if is_palindrome(string, i, j):
        memo[i][j] = 0
        return 0

    curr_min = float('inf')

    for idx in range(i, j):
        if is_palindrome(string, i, idx):
            curr_min = min(curr_min, recur_min_cut(string, idx + 1, j, memo))

    memo[i][j] = curr_min + 1

    return memo[i][j]


def is_palindrome(string, i, j):
    while i < j and string[i] == string[j]:
        i += 1
        j -= 1

    return i >= j

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
