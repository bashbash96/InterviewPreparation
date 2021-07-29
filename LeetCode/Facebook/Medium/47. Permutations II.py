"""
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.



Example 1:

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
Example 2:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]


Constraints:

1 <= nums.length <= 8
-10 <= nums[i] <= 10
"""


class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        counts = Counter(nums)

        res = []

        perms(nums, counts, 0, [], res)

        return res

    # time O(n! * n)
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
