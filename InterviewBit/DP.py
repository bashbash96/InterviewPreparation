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
Asked in:  
Google
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
