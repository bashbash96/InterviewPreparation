# -----------------------------------------------------------------------

""" REVISE
Given a string S and a string T, find the minimum window in S which will contain all the
characters in T in linear time complexity.
Note that when the count of a character C in T is N, then the count of C in minimum window in S should be at least N.

Example :

S = "ADOBECODEBANC"
T = "ABC"
Minimum window is "BANC"

 Note:
If there is no such window in S that covers all characters in T, return the empty string ''.
If there are multiple such windows, return the first occurring minimum window ( with minimum start index ).

"""

from collections import Counter, defaultdict


class Solution:
    # @param A : string
    # @param B : string
    # @return a strings
    def minWindow(self, A, B):
        if len(A) < len(B):
            return ''

        count = Counter(B)
        chars = set(B)
        left_idx = 0
        start_idx = -1
        min_len = float('inf')
        curr_count = defaultdict(int)
        counter = 0
        for i in range(len(A)):
            curr_count[A[i]] += 1
            if A[i] in chars and curr_count[A[i]] <= count[A[i]]:
                counter += 1
            if counter == len(B):
                while left_idx < len(A) and (A[left_idx] not in count or curr_count[A[left_idx]] > count[A[left_idx]]):
                    if curr_count[A[left_idx]] > count[A[left_idx]]:
                        curr_count[A[left_idx]] -= 1
                    left_idx += 1
                if i - left_idx + 1 < min_len:
                    min_len = i - left_idx + 1
                    start_idx = left_idx

        if start_idx == -1:
            return ''

        return A[start_idx: start_idx + min_len]

    # time O(n + m)
    # space O(n + m)

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
