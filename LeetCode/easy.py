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
"""
121. Best Time to Buy and Sell Stock

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in 
the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

"""


class Solution:
    def maxProfit(self, prices):
        if len(prices) == 0:
            return 0

        min_val, max_profit = prices[0], 0
        for price in prices:
            max_profit = max(max_profit, price - min_val)
            min_val = min(min_val, price)

        return max_profit

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------

"""
543. Diameter of Binary Tree

Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree 
is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

Example:
Given a binary tree
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

Note: The length of path between two nodes is represented by the number of edges between them.
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def diameterOfBinaryTree(self, root):
        return diameter(root)[1]

    # time O(n)
    # space O(h)


def diameter(root):
    if not root:
        return 0, 0

    left = diameter(root.left)
    right = diameter(root.right)

    return max(left[0], right[0]) + 1, max(left[1], right[1], left[0] + right[0])


# -----------------------------------------------------------------------

"""
70. Climbing Stairs

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
"""


class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n

        prev1, prev2 = 2, 1

        for _ in range(3, n + 1):
            curr = prev1 + prev2
            prev2 = prev1
            prev1 = curr

        return curr

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

# -----------------------------------------------------------------------
