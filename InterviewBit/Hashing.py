# -----------------------------------------------------------------------

"""
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
"""
Longest Consecutive Sequence

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Example:
Given [100, 4, 200, 1, 3, 2],
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.

Your algorithm should run in O(n) complexity.
"""


class Solution:
    # @param A : tuple of integers
    # @return an integer
    def longestConsecutive(self, A):
        max_length = 0
        visited = set()
        nums = set(A)

        for num in A:
            max_length = max(max_length, get_component_length(num, nums, visited))

        return max_length

    # time O(n)
    # space O(n)


def get_component_length(num, nums, visited):
    if num in visited:
        return 1

    visited.add(num)
    length = 1

    def traverse_component(curr_num, jump):
        nonlocal length
        nonlocal visited
        while curr_num in nums:
            visited.add(curr_num)
            length += 1
            curr_num += jump

    # check right neighbors
    traverse_component(num + 1, 1)
    # curr_num = num + 1
    # while curr_num in nums:
    #     visited.add(curr_num)
    #     curr_num += 1
    #     length += 1

    # check left neighbors
    traverse_component(num - 1, -1)
    # curr_num = num - 1
    # while curr_num in nums:
    #     visited.add(curr_num)
    #     curr_num -= 1
    #     length += 1

    return length

# -----------------------------------------------------------------------
"""
4 Sum

Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

 Note:
Elements in a quadruplet (a,b,c,d) must be in non-descending order. (ie, a ≤ b ≤ c ≤ d)
The solution set must not contain duplicate quadruplets.
Example :
Given array S = {1 0 -1 0 -2 2}, and target = 0
A solution set is:

    (-2, -1, 1, 2)
    (-2,  0, 0, 2)
    (-1,  0, 0, 1)
"""


class Solution:
    # @param A : list of integers
    # @param B : integer
    # @return a list of list of integers
    def fourSum(self, A, B):
        arr = sorted(A)
        target = B

        return kSum(arr, target, 0, 4)

    # time O(n ^ 3)
    # space O(n)


def kSum(arr, target, start, k):
    if k == 2:
        return get_doubles(arr, target, start)

    res = []
    i = start
    while i < len(arr):

        curr = kSum(arr, target - arr[i], i + 1, k - 1)
        for val in curr:
            res.append([arr[i]] + val)

        while i < len(arr) - 1 and arr[i] == arr[i + 1]:
            i += 1
        i += 1

    return res


def get_doubles(arr, target, start):
    end = len(arr) - 1
    res = []

    while start < end:

        curr_sum = arr[start] + arr[end]
        if curr_sum > target:
            end -= 1
        elif curr_sum < target:
            start += 1
        else:
            res.append([arr[start], arr[end]])
            start += 1
            while start < end and arr[start] == arr[start - 1]:
                start += 1

    return res


# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
