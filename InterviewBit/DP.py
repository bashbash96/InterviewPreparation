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
"""
Word Break

Given a string A and a dictionary of words B, determine if A can be segmented into a space-separated sequence of one or more dictionary words.

Input Format:

The first argument is a string, A.
The second argument is an array of strings, B.
Output Format:

Return 0 / 1 ( 0 for false, 1 for true ) for this problem.
Constraints:

1 <= len(A) <= 6500
1 <= len(B) <= 10000
1 <= len(B[i]) <= 20
Examples:

Input 1:
    A = "myinterviewtrainer",
    B = ["trainer", "my", "interview"]

Output 1:
    1

Explanation 1:
    Return 1 ( corresponding to true ) because "myinterviewtrainer" can be segmented as "my interview trainer".
    
Input 2:
    A = "a"
    B = ["aaa"]

Output 2:
    0

Explanation 2:
    Return 0 ( corresponding to false ) because "a" cannot be segmented as "aaa".
"""

from collections import defaultdict


class Solution:
    # @param A : string
    # @param B : list of strings
    # @return an integer
    def wordBreak(self, A, B):
        memo = {}
        return 1 if can_build_it(A, 0, len(A) - 1, set(B), memo) else 0

    # time O(n^2)
    # space O(n)


def can_build_it(sentence, start, end, words, memo):
    if start > end:
        return False

    if sentence[start: end + 1] in words:
        return True

    if (start, end) in memo:
        return memo[(start, end)]

    for k in range(start, end + 1):
        if sentence[start: k + 1] in words:
            if can_build_it(sentence, k + 1, end, words, memo):
                memo[(start, end)] = True
                return True

    memo[(start, end)] = False
    return False


# -----------------------------------------------------------------------
"""
Increasing Path in Matrix

Problem Description

Given a 2D integer matrix A of size N x M.

From A[i][j] you can move to A[i+1][j], if A[i+1][j] > A[i][j], or can move to A[i][j+1] if A[i][j+1] > A[i][j].

The task is to find and output the longest path length if we start from (0, 0).

NOTE:

If there doesn't exist a path return -1.


Problem Constraints
1 <= N, M <= 103

1 <= A[i][j] <= 108



Input Format
First and only argument is an 2D integer matrix A of size N x M.



Output Format
Return a single integer denoting the length of longest path in the matrix if no such path exists return -1.



Example Input
Input 1:

 A = [  [1, 2]
        [3, 4]
     ]
Input 2:

 A = [  [1, 2, 3, 4]
        [2, 2, 3, 4]
        [3, 2, 3, 4]
        [4, 5, 6, 7]
     ]


Example Output
Output 1:

 3
Output 2:

 7
"""


class Solution:
    # @param A : list of list of integers
    # @return an integer
    def solve(self, A):
        arr = A
        n = len(arr)
        m = len(arr[0])

        lp = [[1 for _ in range(m)] for _ in range(n)]

        for col in range(1, m):
            if arr[0][col] > arr[0][col - 1] and lp[0][col - 1] != -1:
                lp[0][col] = lp[0][col - 1] + 1
            else:
                lp[0][col] = -1

        for row in range(1, n):
            if arr[row][0] > arr[row - 1][0] and lp[row - 1][0] != -1:
                lp[row][0] = lp[row - 1][0] + 1
            else:
                lp[row][0] = -1

        for row in range(1, n):
            for col in range(1, m):

                top = arr[row - 1][col]
                left = arr[row][col - 1]
                if arr[row][col] > top and lp[row - 1][col] != -1:
                    lp[row][col] = max(lp[row][col], lp[row - 1][col] + 1)
                else:
                    lp[row][col] = -1

                if arr[row][col] > left and lp[row][col - 1] != -1:
                    lp[row][col] = max(lp[row][col], lp[row][col - 1] + 1)
                else:
                    lp[row][col] = -1

        return lp[n - 1][m - 1]

    # time O(n * m)
    # space O(n * m)


# -----------------------------------------------------------------------
"""
Sub Matrices with sum Zero
Asked in:  
Google
Problem Setter: mihai.gheorghe Problem Tester: sneh_gupta
Given a 2D matrix, find the number non-empty sub matrices, such that the sum of the elements inside the sub matrix is equal to 0. (note: elements might be negative).

Example:

Input

-8 5  7
3  7 -8
5 -8  9
Output
2

Explanation
-8 5 7
3 7 -8
5 -8 9

-8 5 7
3 7 -8
5 -8 9
"""

from collections import defaultdict


class Solution:
    # @param A : list of list of integers
    # @return an integer
    def solve(self, matrix):

        if not matrix:
            return 0
        target = 0
        n = len(matrix)
        m = len(matrix[0])

        pre_sum = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

        for row in range(1, n + 1):
            for col in range(1, m + 1):
                top = pre_sum[row - 1][col]
                left = pre_sum[row][col - 1]
                diagonal = pre_sum[row - 1][col - 1]
                pre_sum[row][col] = top + left - diagonal + matrix[row - 1][col - 1]

        count = 0
        for row in range(1, n + 1):

            for curr_row in range(row, n + 1):

                h_map = defaultdict(int)
                h_map[0] = 1

                for col in range(1, m + 1):
                    curr_sum = pre_sum[curr_row][col] - pre_sum[row - 1][col]

                    count += h_map[curr_sum - target]

                    h_map[curr_sum] += 1

        return count

    # time O(n^2 * m)
    # space O(n * m)


# -----------------------------------------------------------------------
"""
Interleaving Strings

Given A, B, C, find whether C is formed by the interleaving of A and B.

Input Format:*

The first argument of input contains a string, A.
The second argument of input contains a string, B.
The third argument of input contains a string, C.
Output Format:

Return an integer, 0 or 1:
    => 0 : False
    => 1 : True
Constraints:

1 <= length(A), length(B), length(C) <= 150
Examples:

Input 1:
    A = "aabcc"
    B = "dbbca"
    C = "aadbbcbcac"

Output 1:
    1
    
Explanation 1:
    "aa" (from A) + "dbbc" (from B) + "bc" (from A) + "a" (from B) + "c" (from A)

Input 2:
    A = "aabcc"
    B = "dbbca"
    C = "aadbbbaccc"

Output 2:
    0

Explanation 2:
    It is not possible to get C by interleaving A and B.
"""


class Solution:
    # @param A : string
    # @param B : string
    # @param C : string
    # @return an integer
    def isInterleave(self, A, B, C):
        if can_generate(A, B, C, 0, 0, 0, {}):
            return 1

        return 0

    # time O(n + m)
    # space O(n + m)


def can_generate(st1, st2, target, p1, p2, p3, memo):
    if p3 == len(target):
        return True

    if (p1, p2, p3) in memo:
        return memo[(p1, p2, p3)]

    res1 = False
    if p1 < len(st1) and st1[p1] == target[p3]:
        res1 = can_generate(st1, st2, target, p1 + 1, p2, p3 + 1, memo)

    if res1:
        memo[(p1, p2, p3)] = res1
        return True

    res2 = False
    if p2 < len(st2) and st2[p2] == target[p3]:
        res2 = can_generate(st1, st2, target, p1, p2 + 1, p3 + 1, memo)

    if res2:
        memo[(p1, p2, p3)] = res2
        return True

    memo[(p1, p2, p3)] = False

    return False


# -----------------------------------------------------------------------
"""
Longest valid Parentheses

Given a string A containing just the characters ’(‘ and ’)’.

Find the length of the longest valid (well-formed) parentheses substring.



Input Format:

The only argument given is string A.
Output Format:

Return the length of the longest valid (well-formed) parentheses substring.
Constraints:

1 <= length(A) <= 750000
For Example

Input 1:
    A = "(()"
Output 1:
    2
    Explanation 1:
        The longest valid parentheses substring is "()", which has length = 2.

Input 2:
    A = ")()())"
Output 2:
    4
    Explanation 2:
        The longest valid parentheses substring is "()()", which has length = 4.
"""


class Solution:
    # @param A : string
    # @return an integer
    def longestValidParentheses(self, A):

        res = get_max(A)

        left, right = 0, 0

        for c in A[::-1]:
            if c == ')':
                right += 1
            else:
                left += 1

            if left == right:
                res = max(res, left + right)

            if left > right:
                left = 0
                right = 0

        return res

    # time O(n)
    # space O(1)


def get_max(A):
    left, right = 0, 0

    res = 0
    for c in A:
        if c == '(':
            left += 1
        else:
            right += 1

        if right == left:
            res = max(res, left + right)

        if right > left:
            right = 0
            left = 0

    return res

# -----------------------------------------------------------------------
"""
Word Break II

Given a string A and a dictionary of words B, add spaces in A to construct a sentence where each word is a valid dictionary word.

Return all such possible sentences.

Note : Make sure the strings are sorted in your result.

Input Format:

The first argument is a string, A.
The second argument is an array of strings, B.
Output Format:

Return a vector of strings representing the answer as described in the problem statement.
Constraints:

1 <= len(A) <= 50
1 <= len(B) <= 25
1 <= len(B[i]) <= 20
Examples:

Input 1:
    A = "b"
    B = ["aabbb"]

Output 1:
    []

Input 1:
    A = "catsanddog",
    B = ["cat", "cats", "and", "sand", "dog"]

Output 1:
    ["cat sand dog", "cats and dog"]
"""


class Solution:
    # @param A : string
    # @param B : list of strings
    # @return a list of strings
    def wordBreak(self, A, B):
        return generate_valid_sentences(A, set(B), 0, {})

    # time O(2^n * n)
    # space O(n)


def generate_valid_sentences(string, words, idx, memo):
    if idx in memo:
        return memo[idx]

    if idx == len(string):
        return []

    memo[idx] = []
    for end in range(idx + 1, len(string) + 1):

        curr = string[idx:end]
        if curr in words:
            rest = generate_valid_sentences(string, words, end, memo)

            if rest:
                for sentence in rest:
                    memo[idx].append(curr + ' ' + sentence)
            elif end == len(string):
                memo[idx].append(curr)

    return memo[idx]

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
