"""
Given an array nums of integers, return the length of the longest arithmetic subsequence in nums.

Recall that a subsequence of an array nums is a list nums[i1], nums[i2], ..., nums[ik] with 0 <= i1 < i2 < ... < ik <= nums.length - 1, and that a sequence seq is arithmetic if seq[i+1] - seq[i] are all the same value (for 0 <= i < seq.length - 1).



Example 1:

Input: nums = [3,6,9,12]
Output: 4
Explanation:
The whole array is an arithmetic sequence with steps of length = 3.
Example 2:

Input: nums = [9,4,7,2,10]
Output: 3
Explanation:
The longest arithmetic subsequence is [4,7,10].
Example 3:

Input: nums = [20,1,15,3,10,5,8]
Output: 4
Explanation:
The longest arithmetic subsequence is [20,15,10,5].


Constraints:

2 <= nums.length <= 1000
0 <= nums[i] <= 500
"""

from collections import defaultdict


class Solution(object):
    def longestArithSeqLength(self, A):
        """
        :type A: List[int]
        :rtype: int
        """

        arith_length = defaultdict()
        n = len(A)
        res = 0
        for i in range(n):
            curr = defaultdict()
            for j in range(i):
                diff = A[i] - A[j]

                if diff in arith_length[j]:
                    curr[diff] = arith_length[j][diff] + 1
                else:
                    curr[diff] = 2

            arith_length[i] = curr
            if curr:
                res = max(res, max(curr.values()))

        return res

    # time O(n^2)
    # space O(n)
