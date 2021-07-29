"""
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.



Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]


Constraints:

1 <= nums.length <= 6
-10 <= nums[i] <= 10
All the integers of nums are unique.
"""

from collections import Counter


class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        counts = Counter(nums)

        res = []

        perms(nums, counts, 0, [], res)

        return res

    # time O(n!)
    # space O(n)


def perms(nums, counts, idx, curr, res):
    if idx == len(nums):
        res.append(curr[:])
        return

    for num in counts:
        if counts[num] > 0:
            counts[num] -= 1
            perms(nums, counts, idx + 1, curr + [num], res)
            counts[num] += 1
