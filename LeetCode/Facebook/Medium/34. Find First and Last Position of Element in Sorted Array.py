"""
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.


Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
Example 3:

Input: nums = [], target = 0
Output: [-1,-1]


Constraints:

0 <= nums.length <= 105
-109 <= nums[i] <= 109
nums is a non-decreasing array.
-109 <= target <= 109
"""


class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        left = get_limit(nums, target, 'left')
        right = get_limit(nums, target, 'right')

        return [left, right]

    # time O(log(n))
    # space O(1)


def get_limit(nums, target, limit):
    start, end = 0, len(nums) - 1
    res = -1

    while start <= end:
        mid = (start + end) // 2

        if target < nums[mid]:
            end = mid - 1
        elif target > nums[mid]:
            start = mid + 1
        else:
            res = mid
            if limit == 'left':
                end = mid - 1
            else:
                start = mid + 1

    return res
