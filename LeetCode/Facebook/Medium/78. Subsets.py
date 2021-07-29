"""
Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.



Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
Example 2:

Input: nums = [0]
Output: [[],[0]]


Constraints:

1 <= nums.length <= 10
-10 <= nums[i] <= 10
All the numbers of nums are unique.
"""


class Solution:
    def subsets(self, nums):

        if not nums:
            return [[]]

        res = [[]]
        for num in nums:
            curr_list = []
            for list_ in res:
                curr_list.append(list_[:] + [num])
            res.extend(curr_list)

        return res

    # time O(2^n * n)
    # space O(2^n * n)
