import collections
import sys

# -----------------------------------------------------------------------
"""
525. Contiguous Array

Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.

Example 1:
Input: [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with equal number of 0 and 1.
"""


class Solution:
    def findMaxLength(self, nums):
        map = {}
        count, maxLen = 0, 0
        map[0] = -1
        for i in range(len(nums)):
            count += (1 if nums[i] == 1 else -1)
            if count in map:
                maxLen = max(maxLen, i - map[count])
            else:
                map[count] = i

        return maxLen

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
300. Longest Increasing Subsequence

Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:

Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
Note:

There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.
Follow up: Could you improve it to O(n log n) time complexity?
"""


class Solution:
    def lengthOfLIS(self, nums):

        memo = [1] * len(nums)
        LIS = 0
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    memo[i] = max(memo[i], memo[j] + 1)
            LIS = max(LIS, memo[i])

        return LIS

        # time O(n^2)
        # space O(n)


# -----------------------------------------------------------------------
"""
238. Product of Array Except Self

Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of 
all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Constraint: It's guaranteed that the product of the elements of any prefix or suffix of the array 
(including the whole array) fits in a 32 bit integer.

Note: Please solve it without division and in O(n).

Follow up:
Could you solve it with constant space complexity? 
(The output array does not count as extra space for the purpose of space complexity analysis.)
"""


class Solution:
    def productExceptSelf(self, nums):
        n = len(nums)
        left, right = [0] * n, [0] * n

        currProd = 1
        for i in range(n):
            left[i] = currProd
            currProd *= nums[i]

        currProd = 1
        for i in range(n - 1, -1, -1):
            right[i] = currProd
            currProd *= nums[i]

        for i in range(n):
            nums[i] = left[i] * right[i]

        return nums

        # time O(n)
        # space O(n)

    def productExceptSelf(self, nums):
        n = len(nums)

        res = [1] * n
        for i in range(1, n):
            res[i] = res[i - 1] * nums[i - 1]

        right = 1
        for i in range(n - 1, -1, -1):
            res[i] = right * res[i]
            right *= nums[i]

        return res

        # time O(n)
        # space O(1) -> without the result


# -----------------------------------------------------------------------
"""
56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

Example 1:

Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6]
"""


class Solution:
    def merge(self, intervals):
        if len(intervals) < 2:
            return intervals
        intervals.sort(key=lambda pair: pair[0])
        res = []
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            last = res[-1]
            if last[1] >= intervals[i][0]:
                res[-1][1] = max(intervals[i][1], last[1])
            else:
                res.append(intervals[i])

        return res

        # time O(n*log(n))
        # space O(n)


# -----------------------------------------------------------------------
"""
416. Partition Equal Subset Sum

Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note:

Each of the array element will not exceed 100.
The array size will not exceed 200.
 

Example 1:

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
"""


class Solution:
    def canPartition(self, nums):
        if len(nums) < 2:
            return False

        Sum = sum(nums)
        if Sum % 2 != 0:
            return False
        memo = [[None for j in range((Sum // 2) + 1)] for i in range(len(nums))]
        return self.subSetWithTargetSum(nums, 0, 0, Sum // 2, memo)

        # time O(n*s)
        # space O(n*s)

    def subSetWithTargetSum(self, nums, curr, idx, target, memo):
        if curr == target:
            return True

        if curr > target or idx == len(nums):
            return False

        if not memo[idx][curr]:
            taken = self.subSetWithTargetSum(nums, curr + nums[idx], idx + 1, target, memo)
            if taken:
                memo[idx][curr] = taken
                return taken
            notTaken = self.subSetWithTargetSum(nums, curr, idx + 1, target, memo)
            memo[idx][curr] = taken or notTaken

        return memo[idx][curr]


class Solution:
    def canPartition(self, nums):
        if len(nums) < 2:
            return False

        Sum = sum(nums)
        if Sum % 2 != 0:
            return False

        memo = [[True for j in range(len(nums) + 1)] for i in range(Sum // 2 + 1)]

        Sum = (Sum // 2) + 1

        for i in range(1, Sum):
            memo[i][0] = False

        for s in range(1, Sum):
            for i in range(1, len(nums) + 1):
                currNum = nums[i - 1]
                if s >= nums[i - 1]:
                    memo[s][i] = memo[s][i - 1] or memo[s - nums[i - 1]][i - 1]
                else:
                    memo[s][i] = memo[s][i - 1]

        return memo[Sum - 1][len(nums)]

        # time O(n*s)
        # space O(n*s)


# -----------------------------------------------------------------------
"""
678. Valid Parenthesis String

Given a string containing only three types of characters: '(', ')' and '*', write a function to check whether this 
string is valid. We define the validity of a string by these rules:

Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string.
An empty string is also valid.
Example 1:
Input: "()"
Output: True
Example 2:
Input: "(*)"
Output: True
"""


class Solution:
    def checkValidString(self, s: str) -> bool:

        balance = 0
        for c in s:
            if c == ')':
                balance -= 1
            else:
                balance += 1

            if balance < 0:
                return False

        balance = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '(':
                balance -= 1
            else:
                balance += 1

            if balance < 0:
                return False

        return True

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
146. LRU Cache

Design and implement a data structure for Least Recently Used (LRU) cache. 
It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, 
it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a positive capacity.

Follow up:
Could you do both operations in O(1) time complexity?

Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4

"""


class Node:
    def __init__(self, key, data):
        self.data = data
        self.key = key
        self.next = None
        self.prev = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {}
        self.head = None
        self.tail = None

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        resultData = self.map[key].data
        if len(self.map) == 1:
            return resultData
        self.remove(key)
        self.add(key, resultData)
        return resultData

        # time O(1)
        # space O(1)

    def put(self, key: int, value: int) -> None:

        if key in self.map:
            self.remove(key)
            self.add(key, value)
            return
        if len(self.map) < self.capacity:
            self.add(key, value)
            return

        self.remove(self.tail.key)
        self.add(key, value)

        # time O(1)
        # space O(1)

    def add(self, key, value):
        node = Node(key, value)
        self.map[key] = node
        if not self.head:
            self.head = self.tail = node
            return

        node.next = self.head
        self.head.prev = node
        self.head = node

    def remove(self, key):
        if key not in self.map:
            return
        currNode = self.map[key]
        del self.map[key]

        if self.head == currNode and self.tail == currNode:
            self.head = self.tail = None
            return

        currNode.prev.next = currNode.next

        if self.tail == currNode:
            self.tail = currNode.prev
        else:
            if currNode.next:
                currNode.next.prev = currNode.prev

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
