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
    def reverseList(self, head):
        # recursive
        # if not head or not head.next:
        #     return head
        #
        # prev = self.reverseList(head.next)
        # head.next.next = head
        # head.next = None
        # return prev

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
"""
938. Range Sum of BST

Given the root node of a binary search tree, return the sum of values of all nodes with a value in the range [low, high].

 

Example 1:


Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
Output: 32
Example 2:


Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
Output: 23
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        return get_sum(root, low, high)


def get_sum(node, low, high):
    if not node:
        return 0

    if node.val < low:
        return get_sum(node.right, low, high)
    elif node.val > high:
        return get_sum(node.left, low, high)

    return get_sum(node.left, low, high) + node.val + get_sum(node.right, low, high)

    # time O(h + (high - low))
    # space O(h)


# -----------------------------------------------------------------------
"""
1021. Remove Outermost Parentheses
Easy

725

790

Add to List

Share
A valid parentheses string is either empty (""), "(" + A + ")", or A + B, where A and B are valid parentheses strings, and + represents string concatenation.  For example, "", "()", "(())()", and "(()(()))" are all valid parentheses strings.

A valid parentheses string S is primitive if it is nonempty, and there does not exist a way to split it into S = A+B, with A and B nonempty valid parentheses strings.

Given a valid parentheses string S, consider its primitive decomposition: S = P_1 + P_2 + ... + P_k, where P_i are primitive valid parentheses strings.

Return S after removing the outermost parentheses of every primitive string in the primitive decomposition of S.

 

Example 1:

Input: "(()())(())"
Output: "()()()"
Explanation: 
The input string is "(()())(())", with primitive decomposition "(()())" + "(())".
After removing outer parentheses of each part, this is "()()" + "()" = "()()()".
Example 2:

Input: "(()())(())(()(()))"
Output: "()()()()(())"
Explanation: 
The input string is "(()())(())(()(()))", with primitive decomposition "(()())" + "(())" + "(()(()))".
After removing outer parentheses of each part, this is "()()" + "()" + "()(())" = "()()()()(())".
Example 3:

Input: "()()"
Output: ""
Explanation: 
The input string is "()()", with primitive decomposition "()" + "()".
After removing outer parentheses of each part, this is "" + "" = "".
"""


class Solution:
    def removeOuterParentheses(self, S: str) -> str:

        curr_group = []
        res = []
        counter = 0
        for c in S:
            curr_group.append(c)
            if c == '(':
                counter += 1
            else:
                counter -= 1

            if counter == 0:
                # remove outermost parentheses
                curr = ''.join(curr_group[1:-1])
                res.append(curr)
                curr_group = []

        return ''.join(res)

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
617. Merge Two Binary Trees

You are given two binary trees root1 and root2.

Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.

Return the merged tree.

Note: The merging process must start from the root nodes of both trees.

 

Example 1:


Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
Output: [3,4,5,5,4,null,7]
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1, root2):

        if not root1 and not root2:
            return None

        if not root1:
            return root2

        if not root2:
            return root1

        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.val += root2.val
        root1.right = self.mergeTrees(root1.right, root2.right)

        return root1

    # time O(n) n : total nodes
    # space O(h)


# -----------------------------------------------------------------------
"""
163. Missing Ranges

You are given an inclusive range [lower, upper] and a sorted unique integer array nums, where all elements are in the inclusive range.

A number x is considered missing if x is in the range [lower, upper] and x is not in nums.

Return the smallest sorted list of ranges that cover every missing number exactly. That is, no element of nums is in any of the ranges, and each missing number is in one of the ranges.

Each range [a,b] in the list should be output as:

"a->b" if a != b
"a" if a == b
 

Example 1:

Input: nums = [0,1,3,50,75], lower = 0, upper = 99
Output: ["2","4->49","51->74","76->99"]
Explanation: The ranges are:
[2,2] --> "2"
[4,49] --> "4->49"
[51,74] --> "51->74"
[76,99] --> "76->99"
Example 2:

Input: nums = [], lower = 1, upper = 1
Output: ["1"]
Explanation: The only missing range is [1,1], which becomes "1".
"""


class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
        curr_low = lower

        res = []
        for num in nums:
            if num == curr_low:
                curr_low += 1
            elif num > curr_low:
                curr_low = add_range(res, num - 1, curr_low)
                if curr_low > upper:
                    break

        if curr_low <= upper:
            add_range(res, upper, curr_low)
        return res

    # time O(n)
    # space O(n)


def add_range(res, upper, lower):
    if upper == lower:
        res.append(str(lower))
    else:
        res.append(f'{lower}->{upper}')

    return upper + 2


# -----------------------------------------------------------------------
"""
977. Squares of a Sorted Array

Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.

 

Example 1:

Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
Example 2:

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
"""


class Solution:
    def sortedSquares(self, nums):

        non_negative_idx = get_first_non_negative_idx(nums)
        negative_idx = non_negative_idx - 1
        res = []
        while non_negative_idx < len(nums) and negative_idx >= 0:
            n_g_num = nums[non_negative_idx]
            n_num = nums[negative_idx]
            if n_num * n_num <= n_g_num * n_g_num:
                res.append(n_num * n_num)
                negative_idx -= 1
            else:
                res.append(n_g_num * n_g_num)
                non_negative_idx += 1

        while non_negative_idx < len(nums):
            n_g_num = nums[non_negative_idx]
            res.append(n_g_num * n_g_num)
            non_negative_idx += 1

        while negative_idx >= 0:
            n_num = nums[negative_idx]
            res.append(n_num * n_num)
            negative_idx -= 1

        return res

    # time O(n)
    # space O(n)


def get_first_non_negative_idx(nums):
    i = 1
    while i < len(nums):
        if nums[i] >= 0:
            return i
        i += 1

    return i


# -----------------------------------------------------------------------
"""
929. Unique Email Addresses

Every email consists of a local name and a domain name, separated by the @ sign.

For example, in alice@leetcode.com, alice is the local name, and leetcode.com is the domain name.

Besides lowercase letters, these emails may contain '.'s or '+'s.

If you add periods ('.') between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name.  For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.  (Note that this rule does not apply for domain names.)

If you add a plus ('+') in the local name, everything after the first plus sign will be ignored. This allows certain emails to be filtered, for example m.y+name@email.com will be forwarded to my@email.com.  (Again, this rule does not apply for domain names.)

It is possible to use both of these rules at the same time.

Given a list of emails, we send one email to each address in the list.  How many different addresses actually receive mails? 

 

Example 1:

Input: ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
Output: 2
Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails
"""


class Solution:
    def numUniqueEmails(self, emails):
        unique_emails = set()
        for email in emails:
            unique_emails.add(process_email(email))

        return len(unique_emails)

    # time O(n*m)
    # space O(n*m)


def process_email(email):
    local_name, domain_name = email.split('@')
    if '+' in local_name:
        local_name = local_name[:local_name.index('+')]
    if '.' in local_name:
        local_name = local_name.replace('.', '')

    return local_name + '@' + domain_name


# -----------------------------------------------------------------------
"""
226. Invert Binary Tree

Invert a binary tree.

Example:

Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
Trivia:
This problem was inspired by this original tweet by Max Howell:

Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so f*** off.
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root):
        if not root:
            return root

        curr_queue = [root]

        while len(curr_queue) > 0:
            curr_node = curr_queue.pop(0)
            curr_node.left, curr_node.right = curr_node.right, curr_node.left
            if curr_node.left:
                curr_queue.append(curr_node.left)
            if curr_node.right:
                curr_queue.append(curr_node.right)

        return root

    # time O(n)
    # space O(width)


# -----------------------------------------------------------------------
"""
706. Design HashMap

Design a HashMap without using any built-in hash table libraries.

To be specific, your design should include these functions:

put(key, value) : Insert a (key, value) pair into the HashMap. If the value already exists in the HashMap, update the value.
get(key): Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
remove(key) : Remove the mapping for the value key if this map contains the mapping for the key.

Example:

MyHashMap hashMap = new MyHashMap();
hashMap.put(1, 1);          
hashMap.put(2, 2);         
hashMap.get(1);            // returns 1
hashMap.get(3);            // returns -1 (not found)
hashMap.put(2, 1);          // update the existing value
hashMap.get(2);            // returns 1 
hashMap.remove(2);          // remove the mapping for 2
hashMap.get(2);            // returns -1 (not found) 
"""


class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 2069
        self.table = [[] for _ in range(self.size)]

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        hash_key = self.hash(key)
        bucket_list = self.table[hash_key]
        if bucket_list == []:
            self.table[hash_key] = [(key, value)]
        else:
            self.insert_into_bucket(bucket_list, key, value)

        # time O(k) - k the largest bucket

    def insert_into_bucket(self, bucket, key, value):

        key_idx = self.search_for_key(bucket, key)
        if key_idx != -1:
            bucket[key_idx] = (key, value)
        else:
            bucket.append((key, value))

    def search_for_key(self, bucket, key):

        for idx, pair in enumerate(bucket):
            if pair[0] == key:
                return idx
        return -1

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """

        hash_key = self.hash(key)
        bucket = self.table[hash_key]
        if bucket == []:
            return -1

        key_idx = self.search_for_key(bucket, key)
        if key_idx != -1:
            return bucket[key_idx][1]

        return -1

    # time O(k) - k the largest bucket

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        hash_key = self.hash(key)
        bucket = self.table[hash_key]
        key_idx = self.search_for_key(bucket, key)
        if key_idx != -1:
            bucket[key_idx], bucket[-1] = bucket[-1], bucket[key_idx]
            bucket.pop()

    # time O(k) - k the largest bucket

    def hash(self, key):
        return key % 2069


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

# -----------------------------------------------------------------------
"""
13. Roman to Integer

Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
Example 2:

Input: s = "IV"
Output: 4
Example 3:

Input: s = "IX"
Output: 9
Example 4:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
"""

roman_to_decimal = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}


class Solution:
    def romanToInt(self, s: str) -> int:

        if s == '':
            return 0

        res = 0
        for idx in range(len(s) - 1):
            curr_char, next_char = s[idx], s[idx + 1]
            if compare_vals(curr_char, next_char) >= 0:
                res += roman_to_decimal[curr_char]
            else:
                res -= roman_to_decimal[curr_char]

        res += roman_to_decimal[s[-1]]

        return res

    # time O(n)
    # space O(1)


def compare_vals(val1, val2):
    if roman_to_decimal[val1] > roman_to_decimal[val2]:
        return 1
    elif roman_to_decimal[val1] < roman_to_decimal[val2]:
        return -1

    return 0


# -----------------------------------------------------------------------
"""
733. Flood Fill

An image is represented by a 2-D array of integers, each integer representing the pixel value of the image (from 0 to 65535).

Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill, and a pixel value newColor, "flood fill" the image.

To perform a "flood fill", consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all of the aforementioned pixels with the newColor.

At the end, return the modified image.

Example 1:
Input: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation: 
From the center of the image (with position (sr, sc) = (1, 1)), all pixels connected 
by a path of the same color as the starting pixel are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected
to the starting pixel.
"""


class Solution:
    def floodFill(self, image, sr: int, sc: int, newColor: int):
        if newColor == image[sr][sc]:
            return image

        fill(image, sr, sc, newColor, image[sr][sc])

        return image

    # time O(n * m)
    # space O(n * m)


def is_valid(image, row, col):
    if row < 0 or col < 0 or row >= len(image) or col >= len(image[0]):
        return False

    return True


def get_neighbors(row, col):
    # top right down left
    directions = {'top': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}

    # directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    res = []
    for direction in directions:
        dx, dy = directions[direction]
        res.append((row + dx, col + dy))

    return res


def fill(image, row, col, new_color, old_color):
    if not is_valid(image, row, col) or image[row][col] != old_color:
        return

    image[row][col] = new_color

    for n_row, n_col in get_neighbors(row, col):
        fill(image, n_row, n_col, new_color, old_color)


# -----------------------------------------------------------------------
"""
118. Pascal's Triangle

Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:


 

Example 1:

Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
Example 2:

Input: numRows = 1
Output: [[1]]
"""


class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 1:
            return [[1]]
        if numRows == 0:
            return []
        res = [[1], [1, 1]]
        while numRows > 2:
            prev = res[-1]

            res.append(get_row(prev))

            numRows -= 1

        return res

    # time O(n^2)
    # space O(n^2)


def get_row(prev_row):
    new_row = [1]

    for i in range(1, len(prev_row)):
        new_row.append(prev_row[i] + prev_row[i - 1])

    new_row.append(1)

    return new_row


# -----------------------------------------------------------------------
"""
37. Reorder Data in Log Files

You are given an array of logs. Each log is a space-delimited string of words, where the first word is the identifier.

There are two types of logs:

Letter-logs: All words (except the identifier) consist of lowercase English letters.
Digit-logs: All words (except the identifier) consist of digits.
Reorder these logs so that:

The letter-logs come before all digit-logs.
The letter-logs are sorted lexicographically by their contents. If their contents are the same, then sort them lexicographically by their identifiers.
The digit-logs maintain their relative ordering.
Return the final order of the logs.

 

Example 1:

Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]
Explanation:
The letter-log contents are all different, so their ordering is "art can", "art zero", "own kit dig".
The digit-logs have a relative order of "dig1 8 1 5 1", "dig2 3 6".
"""


class Solution:
    def reorderLogFiles(self, logs):
        letters, digits = [], []

        for log in logs:
            if log.split()[1].isalpha():
                letters.append(log)
            else:
                digits.append(log)

        letters = [process_log(log) for log in letters]
        letters.sort()
        letters = [log[1] + ' ' + log[0] for log in letters]

        return letters + digits

    # time O(n + k*log(k)*m) - n: number of logs, k: number of letters logs, m the longest log in letters
    # space O(n)


def process_log(log):
    splitted = log.split()
    key, other = splitted[0], splitted[1:]

    return (' '.join(other), key)


# -----------------------------------------------------------------------
"""
100. Same Tree

Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

 

Example 1:


Input: p = [1,2,3], q = [1,2,3]
Output: true
Example 2:


Input: p = [1,2], q = [1,null,2]
Output: false
Example 3:


Input: p = [1,2,1], q = [1,1,2]
Output: false

"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True

        if not p or not q:
            return False

        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    # time O(min(n, m))
    # space O(h)


# -----------------------------------------------------------------------
"""
844. Backspace String Compare

Given two strings S and T, return if they are equal when both are typed into empty text editors. # means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

Example 1:

Input: S = "ab#c", T = "ad#c"
Output: true
Explanation: Both S and T become "ac".
Example 2:

Input: S = "ab##", T = "c#d#"
Output: true
Explanation: Both S and T become "".
Example 3:

Input: S = "a##c", T = "#a#c"
Output: true
Explanation: Both S and T become "c".
Example 4:

Input: S = "a#c", T = "b"
Output: false
Explanation: S becomes "c" while T becomes "b".
Note:

1 <= S.length <= 200
1 <= T.length <= 200
S and T only contain lowercase letters and '#' characters.
Follow up:

Can you solve it in O(N) time and O(1) space?
"""


class Solution(object):
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """

        return get_final_str(S) == get_final_str(T)

    # time O(n + m)
    # space O(n + m)


def get_final_str(string):
    res = []
    skip_count = 0
    for i in range(len(string) - 1, -1, -1):
        if string[i] == '#':
            skip_count += 1
        elif skip_count == 0:
            res.append(string[i])
        else:
            skip_count -= 1

    return ''.join(res)


class Solution(object):
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """

        s_p, t_p = len(S) - 1, len(T) - 1
        skip_s, skip_t = 0, 0

        while s_p >= 0 or t_p >= 0:

            # reach the first valid char in S
            s_p = get_valid_idx(S, s_p, skip_s)

            # reach the first valid char in T
            t_p = get_valid_idx(T, t_p, skip_t)

            if t_p >= 0 and s_p >= 0 and S[s_p] != T[t_p]:
                return False

            # if one reached the end and the other didn't
            if (s_p >= 0) != (t_p >= 0):
                return False

            s_p -= 1
            t_p -= 1

        return True

    # time O(n + m)
    # space O(1)


def get_valid_idx(string, curr_idx, skip):
    while curr_idx >= 0:
        if string[curr_idx] == '#':
            skip += 1
        elif skip > 0:
            skip -= 1
        else:
            break
        curr_idx -= 1

    return curr_idx


# -----------------------------------------------------------------------
"""
387. First Unique Character in a String

Given a string, find the first non-repeating character in it and return its index. If it doesn't exist, return -1.

Examples:

s = "leetcode"
return 0.

s = "loveleetcode"
return 2.
"""

from collections import Counter


class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """

        counts = Counter(s)

        for i in range(len(s)):
            if counts[s[i]] == 1:
                return i

        return -1

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
485. Max Consecutive Ones

Given a binary array, find the maximum number of consecutive 1s in this array.

Example 1:
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.
Note:

The input array will only contain 0 and 1.
The length of input array is a positive integer and will not exceed 10,000
"""


class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curr_count, max_count = 0, 0

        for num in nums:
            if num == 1:
                curr_count += 1
            else:
                max_count = max(max_count, curr_count)
                curr_count = 0

        return max(max_count, curr_count)

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
157. Read N Characters Given Read4

Given a file and assume that you can only read the file using a given method read4, implement a method to read n characters.

 

Method read4:

The API read4 reads 4 consecutive characters from the file, then writes those characters into the buffer array buf4.

The return value is the number of actual characters read.

Note that read4() has its own file pointer, much like FILE *fp in C.

Definition of read4:

    Parameter:  char[] buf4
    Returns:    int

Note: buf4[] is destination not source, the results from read4 will be copied to buf4[]
Below is a high level example of how read4 works:



File file("abcde"); // File is "abcde", initially file pointer (fp) points to 'a'
char[] buf4 = new char[4]; // Create buffer with enough space to store characters
read4(buf4); // read4 returns 4. Now buf4 = "abcd", fp points to 'e'
read4(buf4); // read4 returns 1. Now buf4 = "e", fp points to end of file
read4(buf4); // read4 returns 0. Now buf4 = "", fp points to end of file
 

Method read:

By using the read4 method, implement the method read that reads n characters from the file and store it in the buffer array buf. Consider that you cannot manipulate the file directly.

The return value is the number of actual characters read.

Definition of read:

    Parameters:	char[] buf, int n
    Returns:	int

Note: buf[] is destination not source, you will need to write the results to buf[]
"""

"""
The read4 API is already defined for you.

    @param buf4, a list of characters
    @return an integer
    def read4(buf4):

# Below is an example of how the read4 API can be called.
file = File("abcdefghijk") # File is "abcdefghijk", initially file pointer (fp) points to 'a'
buf4 = [' '] * 4 # Create buffer with enough space to store characters
read4(buf4) # read4 returns 4. Now buf = ['a','b','c','d'], fp points to 'e'
read4(buf4) # read4 returns 4. Now buf = ['e','f','g','h'], fp points to 'i'
read4(buf4) # read4 returns 3. Now buf = ['i','j','k',...], fp points to end of file
"""


class Solution(object):
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Number of characters to read (int)
        :rtype: The number of actual characters read (int)
        """
        ret_val, buff4, curr_idx = 0, ['', '', '', ''], 0

        while curr_idx < n:
            ret_val = read4(buff4)
            if ret_val == 0:
                break

            end_idx = min(n, curr_idx + ret_val)
            ret_idx = 0
            while curr_idx < end_idx:
                buf[curr_idx] = buff4[ret_idx]
                ret_idx += 1
                curr_idx += 1

        return curr_idx

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
1207. Unique Number of Occurrences

Given an array of integers arr, write a function that returns true if and only if the number of occurrences of each value in the array is unique.

 

Example 1:

Input: arr = [1,2,2,1,1,3]
Output: true
Explanation: The value 1 has 3 occurrences, 2 has 2 and 3 has 1. No two values have the same number of occurrences.
Example 2:

Input: arr = [1,2]
Output: false
"""

from collections import Counter


class Solution(object):
    def uniqueOccurrences(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """

        count = Counter(arr)
        seen = set()
        for val in count:
            if count[val] in seen:
                return False
            seen.add(count[val])

        return True

    # time O(n)
    # space O(n)

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
