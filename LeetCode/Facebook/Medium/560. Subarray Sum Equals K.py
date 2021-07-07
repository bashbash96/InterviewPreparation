"""
Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.



Example 1:

Input: nums = [1,1,1], k = 2
Output: 2
Example 2:

Input: nums = [1,2,3], k = 3
Output: 2


Constraints:

1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107
"""

from collections import defaultdict


class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        curr_sum, counter, sum_map = 0, 0, defaultdict(int)

        sum_map[0] = 1

        for num in nums:
            curr_sum += num
            diff = curr_sum - k
            if diff in sum_map:
                counter += sum_map[diff]

            sum_map[curr_sum] += 1

        return counter

    # time O(n)
    # space O(n)
