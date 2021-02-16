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
"""
101. Symmetric Tree

Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
 

But the following [1,2,2,null,3,null,3] is not:

    1
   / \
  2   2
   \   \
   3    3
 

Follow up: Solve it both recursively and iteratively.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# iterative:
# class Solution:
#     def isSymmetric(self, root: TreeNode) -> bool:
#         if not root:
#             return True
#
#         q = [root, root]
#         while len(q) > 0:
#             node1, node2 = q.pop(0), q.pop(0)
#
#             if not node1 and not node2:
#                 continue
#             if not node1 or not node2:
#                 return False
#             if node1.val != node2.val:
#                 return False
#
#             q.append(node1.left)
#             q.append(node2.right)
#             q.append(node1.right)
#             q.append(node2.left)
#
#         return True
#
#     # time O(n)
#     # space O(n)


# recursive:
class Solution:
    def isSymmetric(self, root):
        if not root:
            return True

        return is_sym(root, root)

    # time O(n)
    # space O(n)


def is_sym(node1, node2):
    if not node1 and not node2:
        return True
    if not node1 or not node2:
        return False
    if node1.val != node2.val:
        return False

    return is_sym(node1.left, node2.right) and is_sym(node1.right, node2.left)


# -----------------------------------------------------------------------
"""
53. Maximum Subarray

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

 

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Example 2:

Input: nums = [1]
Output: 1
Example 3:

Input: nums = [0]
Output: 0
Example 4:

Input: nums = [-1]
Output: -1
Example 5:

Input: nums = [-100000]
Output: -100000
 

Constraints:

1 <= nums.length <= 3 * 104
-105 <= nums[i] <= 105
"""


class Solution:
    def maxSubArray(self, nums):

        if len(nums) <= 1:
            return sum(nums)

        final_max, curr_sum = float('-inf'), 0

        for num in nums:
            curr_sum = max(curr_sum + num, num)
            final_max = max(final_max, curr_sum)

        return final_max

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
1. Two Sum

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
"""


class Solution:
    def twoSum(self, nums, target):
        hash_map = {}
        for i in range(len(nums)):
            num = nums[i]
            diff = target - num
            if diff in hash_map and hash_map[diff] != i:
                return [hash_map[diff], i]

            hash_map[num] = i

        return []

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
346. Moving Average from Data Stream

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the MovingAverage class:

MovingAverage(int size) Initializes the object with the size of the window size.
double next(int val) Returns the moving average of the last size values of the stream.
 

Example 1:

Input
["MovingAverage", "next", "next", "next", "next"]
[[3], [1], [10], [3], [5]]
Output
[null, 1.0, 5.5, 4.66667, 6.0]

Explanation
MovingAverage movingAverage = new MovingAverage(3);
movingAverage.next(1); // return 1.0 = 1 / 1
movingAverage.next(10); // return 5.5 = (1 + 10) / 2
movingAverage.next(3); // return 4.66667 = (1 + 10 + 3) / 3
movingAverage.next(5); // return 6.0 = (10 + 3 + 5) / 3
"""

from collections import deque


class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.size = size
        self.stream = deque([])
        self.curr_sum = 0
        self.curr_size = 0

    def next(self, val: int) -> float:
        if self.curr_size < self.size:
            self.curr_size += 1
        else:
            self.curr_sum -= self.stream.popleft()

        self.stream.append(val)
        self.curr_sum += val

        return self.curr_sum / self.curr_size

    # time O(1)
    # space O(size)


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)

# -----------------------------------------------------------------------
"""
359. Logger Rate Limiter

Design a logger system that receives a stream of messages along with their timestamps. Each unique message should 
only be printed at most every 10 seconds (i.e. a message printed at timestamp t will prevent other identical messages 
from being printed until timestamp t + 10).

All messages will come in chronological order. Several messages may arrive at the same timestamp.

Implement the Logger class:

Logger() Initializes the logger object.
bool shouldPrintMessage(int timestamp, string message) Returns true if the message should be printed in the given timestamp, otherwise returns false.
 

Example 1:

Input
["Logger", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage"]
[[], [1, "foo"], [2, "bar"], [3, "foo"], [8, "bar"], [10, "foo"], [11, "foo"]]
Output
[null, true, true, false, false, false, true]

"""


class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """

        self.messages_last_print = {}

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """

        if message not in self.messages_last_print:
            self.messages_last_print[message] = timestamp
            return True

        if timestamp - self.messages_last_print[message] >= 10:
            self.messages_last_print[message] = timestamp
            return True

        return False

    # time O(1)
    # space O(n)


# Your Logger object will be instantiated and called as such:
# obj = Logger()
# param_1 = obj.shouldPrintMessage(timestamp,message)

# -----------------------------------------------------------------------
"""
690. Employee Importance

You are given a data structure of employee information, which includes the employee's unique id, 
their importance value and their direct subordinates' id.

For example, employee 1 is the leader of employee 2, and employee 2 is the leader of employee 3. 
They have importance value 15, 10 and 5, respectively. Then employee 1 has a data structure like [1, 15, [2]], 
and employee 2 has [2, 10, [3]], and employee 3 has [3, 5, []]. Note that although employee 3 is also a subordinate of 
employee 1, the relationship is not direct.

Now given the employee information of a company, and an employee id, you need to return the total importance value 
of this employee and all their subordinates.

Example 1:

Input: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
Output: 11
Explanation:
Employee 1 has importance value 5, and he has two direct subordinates: employee 2 and employee 3. 
They both have importance value 3. So the total importance value of employee 1 is 5 + 3 + 3 = 11.
"""

"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

from collections import deque


class Solution:
    def getImportance(self, employees, id):

        employees_ids = {}
        for employee in employees:
            employees_ids[employee.id] = employee

        q = deque([id])
        total_importance = 0
        while len(q) > 0:
            curr_id = q.popleft()
            curr_employee = employees_ids[curr_id]
            total_importance += curr_employee.importance
            q.extend(curr_employee.subordinates)

        return total_importance

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
66. Plus One

Given a non-empty array of decimal digits representing a non-negative integer, increment one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contains a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

 

Example 1:

Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Example 2:

Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
Example 3:

Input: digits = [0]
Output: [1]
"""


class Solution:
    def plusOne(self, digits):
        if digits == [0]:
            return [1]

        digits.reverse()

        carry = 1
        for i in range(len(digits)):
            curr_sum = digits[i] + carry
            digits[i] = curr_sum % 10
            carry = curr_sum // 10
            if not carry:
                return reversed(digits)

        if carry:
            digits.append(carry)

        return reversed(digits)

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
