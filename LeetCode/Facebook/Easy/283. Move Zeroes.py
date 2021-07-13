"""
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.



Example 1:

Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
Example 2:

Input: nums = [0]
Output: [0]


Constraints:

1 <= nums.length <= 104
-231 <= nums[i] <= 231 - 1


Follow up: Could you minimize the total number of operations done?
"""


class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        n = len(nums)

        idx = 0

        for i in range(n):
            if nums[i] != 0:
                nums[i], nums[idx] = nums[idx], nums[i]
                idx += 1

#         first_zero, first_non_zero = 0, 0

#         while first_non_zero < n and first_zero < n:
#             while first_zero < n and nums[first_zero] != 0:
#                 first_zero += 1

#             if first_non_zero < first_zero:
#                 first_non_zero = first_zero + 1

#             while first_non_zero < n and nums[first_non_zero] == 0:
#                 first_non_zero += 1

#             if first_zero < n and first_non_zero < n:
#                 nums[first_zero], nums[first_non_zero] = nums[first_non_zero], nums[first_zero]
#                 first_zero += 1

# time O(n)
# space O(1)
