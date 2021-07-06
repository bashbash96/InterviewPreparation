"""
Given an integer array nums and an integer k, return true if nums has a continuous subarray of size at least two whose elements sum up to a multiple of k, or false otherwise.

An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.



Example 1:

Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.
Example 2:

Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.
Example 3:

Input: nums = [23,2,6,4,7], k = 13
Output: false


Constraints:

1 <= nums.length <= 105
0 <= nums[i] <= 109
0 <= sum(nums[i]) <= 231 - 1
1 <= k <= 231 - 1
"""

from collections import defaultdict


class Solution(object):
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """

        if len(nums) < 2 or k == 0:
            return False

        mod_idx = defaultdict()
        mod_idx[0] = -1

        curr_sum = 0
        for i, val in enumerate(nums):
            curr_sum = (curr_sum + val) % k

            if curr_sum in mod_idx:
                if i - mod_idx[curr_sum] >= 2:
                    return True
            else:
                mod_idx[curr_sum] = i

        return False

    # time O(n)
    # space O(n)


"""
the size is at least two

the sum is multiple of K! -> mod k = 0


approach 1:
    1. iterate over all subarrays with running sum.
    2. check if the current subarray is multiple of K.

    # time O(n^2)
    # space O(1)

approach 2:
    1. running sum % k and hash map.
    2. save for each running sum its first idxs.
    3. if found running sum that has two idxs with differ by 2 return True
    4. if not found any, return False

    # time O(n)
    # space O(n) 

    [23, 2, 4, 6, 7]
     0   1  2  3  4
     5   1  5  5  0
        
"""
