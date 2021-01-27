import collections
import heapq

# -----------------------------------------------------------------------
"""
198. House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed,
the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and
it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount
of money you can rob tonight without alerting the police.

Example 1:

Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
"""


class Solution:
    def rob(self, nums):
        if len(nums) == 0:
            return 0
        if len(nums) < 3:
            return max(nums)

        prev1 = nums[0]
        prev2 = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            curr = max(prev2, prev1 + nums[i])
            prev1 = prev2
            prev2 = curr

        return prev2

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
697. Degree of an Array

Given a non-empty array of non-negative integers nums, the degree of this array is defined as the maximum frequency 
of any one of its elements.

Your task is to find the smallest possible length of a (contiguous) subarray of nums, that has the same degree as nums.

Example 1:
Input: [1, 2, 2, 3, 1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.
Example 2:
Input: [1,2,2,3,1,4,2]
Output: 6

"""


class Solution:
    def findShortestSubArray(self, nums):
        count = {}
        boundaries = collections.defaultdict(list)
        degree = 0
        res = float('inf')
        for i in range(len(nums)):
            num = nums[i]
            if num not in boundaries:
                boundaries[num] = [i, i]
            else:
                boundaries[num][1] = i

            if num not in count:
                count[num] = 1
            else:
                count[num] += 1

            if count[num] > degree:
                degree = count[num]
                res = boundaries[num][1] - boundaries[num][0] + 1
            elif count[num] == degree:
                currLength = boundaries[num][1] - boundaries[num][0] + 1
                res = min(res, currLength)

        return res

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
136. Single Number

Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

Follow up: Could you implement a solution with a linear runtime complexity and without using extra memory?
"""


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num

        return res

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
206. Reverse Linked List

Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
Follow up:

A linked list can be reversed either iteratively or recursively. Could you implement both?
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:

        if not head:
            return head

        prev, curr = None, head

        while curr:
            next_ = curr.next
            curr.next = prev
            prev = curr
            curr = next_

        return prev

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
283. Move Zeroes

Given an array nums, write a function to move all 0's to the end of it while maintaining the 
relative order of the non-zero elements.

Example:

Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
Note:

You must do this in-place without making a copy of the array.
Minimize the total number of operations
"""


class Solution:
    def moveZeroes(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        firstZero, firstNonZero = 0, 0

        while firstZero < len(nums) and firstNonZero < len(nums):

            while firstZero < len(nums) and nums[firstZero] != 0:
                firstZero += 1

            while firstNonZero < len(nums) and nums[firstNonZero] == 0:
                firstNonZero += 1

            if firstZero < len(nums) and firstNonZero < len(nums):
                if firstZero < firstNonZero:
                    # swap them
                    nums[firstZero], nums[firstNonZero] = nums[firstNonZero], nums[firstZero]
                firstNonZero += 1

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
448. Find All Numbers Disappeared in an Array

Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

Example:

Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]

"""


class Solution:
    def findDisappearedNumbers(self, nums):
        if not nums:
            return []

        for i in range(len(nums)):
            curr_index = abs(nums[i]) - 1
            if nums[curr_index] < 0:
                continue
            nums[curr_index] *= -1

        res = []
        for i in range(len(nums)):
            if nums[i] > 0:
                res.append(i + 1)

        return res

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
21. Merge Two Sorted Lists

Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together 
the nodes of the first two lists.

 
Example 1:

Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:

Input: l1 = [], l2 = []
Output: []

Example 3:

Input: l1 = [], l2 = [0]
Output: [0]
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1, l2):
        res = ListNode('DUMMY')
        curr = res

        p1, p2 = l1, l2

        while p1 and p2:

            if p1.val < p2.val:
                curr.next = p1
                p1 = p1.next
            else:
                curr.next = p2
                p2 = p2.next

            curr = curr.next

        if p1:
            curr.next = p1

        if p2:
            curr.next = p2

        return res.next

    # time O(n)
    # space O(1)

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
