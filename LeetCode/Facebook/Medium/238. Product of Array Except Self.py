"""
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.



Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]


Constraints:

2 <= nums.length <= 105
-30 <= nums[i] <= 30
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.


Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space
for space complexity analysis.)
"""


class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        n = len(nums)
        res = [0] * n
        res[0] = 1

        for i in range(1, n):
            res[i] = res[i - 1] * nums[i - 1]

        right = 1
        for i in range(n - 1, -1, -1):
            res[i] = res[i] * right
            right *= nums[i]

        return res

    # time O(n)
    # space O(1)


#         res = []
#         left_product = get_product(nums)
#         right_product = get_product(nums[::-1])[::-1]

#         for i in range(len(nums)):
#             res.append(left_product[i] * right_product[i])

#         return res

#     # time O(n)
#     # space O(n)


def get_product(nums):
    res = []

    curr_product = 1
    for num in nums:
        res.append(curr_product)
        curr_product *= num

    return res


"""
        [1, 2, 3, 4]

left    [1, 1, 2, 6]
right   [24, 12, 4, 1]

res     [24, 12, 8, 6]


O(1) space
res     [1, 1, 2, 6]

res     [24, 12, 8, 6]
"""
