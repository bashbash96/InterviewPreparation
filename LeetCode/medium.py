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
"""
208. Implement Trie (Prefix Tree)

Implement a trie with insert, search, and startsWith methods.

Example:

Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false
trie.startsWith("app"); // returns true
trie.insert("app");   
trie.search("app");     // returns true
Note:

You may assume that all inputs are consist of lowercase letters a-z.
All inputs are guaranteed to be non-empty strings.

"""


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        curr = self.root
        for c in word:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]
        curr['*'] = True

        # time O(n)
        # space O(n)

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        curr = self.root
        for c in word:
            if c not in curr:
                return False
            curr = curr[c]

        return '*' in curr

        # time O(n)
        # space O(1)

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        curr = self.root
        for c in prefix:
            if c not in curr:
                return False
            curr = curr[c]

        return True

        # time O(n)
        # space O(1)


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

# -----------------------------------------------------------------------
"""
560. Subarray Sum Equals K

Given an array of integers and an integer k, you need to find the total number of continuous subarrays 
whose sum equals to k.

Example 1:
Input:nums = [1,1,1], k = 2
Output: 2
Note:
The length of the array is in range [1, 20,000].
The range of numbers in the array is [-1000, 1000] and the range of the integer k is [-1e7, 1e7].
"""


class Solution:
    def subarraySum(self, nums, k):
        n = len(nums)
        currSum, counter, map = 0, 0, collections.defaultdict(int)
        map[0] = 1
        for num in nums:
            currSum += num
            if currSum - k in map:
                counter += map[currSum - k]
            map[currSum] += 1
        return counter

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
338. Counting Bits

Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate 
the number of 1's in their binary representation and return them as an array.

Example 1:

Input: 2
Output: [0,1,1]
Example 2:

Input: 5
Output: [0,1,1,2,1,2]
"""


class Solution:
    def countBits(self, num):
        counts = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
        if num == 0:
            return [0]
        res = []
        for n in range(num + 1):
            curr_count = 0
            while n > 0:
                curr_count += counts[n & 0x0f]
                n >>= 4
            res.append(curr_count)

        return res

    # time O(n * sizeof(integer) / 4)
    # space O(1)


# -----------------------------------------------------------------------
"""
763. Partition Labels

A string S of lowercase English letters is given. We want to partition this string into as many parts as possible 
so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

 

Example 1:

Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
"""


class Solution:
    def partitionLabels(self, S):
        letters_count = collections.Counter(S)

        curr = []
        res = []
        curr_count = collections.defaultdict(int)
        for c in S:
            curr_count[c] += 1
            curr.append(c)
            if curr_count[c] == letters_count[c]:
                del curr_count[c]

            if len(curr_count) == 0:
                res.append(len(curr))
                curr = []

        return res


# -----------------------------------------------------------------------
"""
406. Queue Reconstruction by Height

You are given an array of people, people, which are the attributes of some people in a queue 
(not necessarily in order). Each people[i] = [hi, ki] represents the ith person of height hi with exactly ki other 
people in front who have a height greater than or equal to hi.

Reconstruct and return the queue that is represented by the input array people. 
The returned queue should be formatted as an array queue, where queue[j] = [hj, kj] is the attributes of the jth person 
in the queue (queue[0] is the person at the front of the queue).
"""


class Solution:
    def reconstructQueue(self, people):
        people.sort(key=lambda x: (x[0], x[1]))
        curr_count = 0
        res = [None for _ in people]
        for val in people:
            curr_count = val[1]
            place_element(res, val, curr_count)

        return res

    # time O(n^2)
    # space O(n)


def place_element(res, val, curr_count):
    for i in range(len(res)):
        if curr_count == 0 and res[i] == None:
            res[i] = val
            return
        elif res[i] == None or res[i][0] >= val[0]:
            curr_count -= 1


# -----------------------------------------------------------------------
"""
94. Binary Tree Inorder Traversal

Given the root of a binary tree, return the inorder traversal of its nodes' values.

Example 1:

Input: root = [1,null,2,3]
Output: [1,3,2]


Example 2:

Input: root = []
Output: []

Example 3:

Input: root = [1]
Output: [1]
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root):
        if not root:
            return []

        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    # time O(n)
    # space O(h)


# -----------------------------------------------------------------------
"""
22. Generate Parentheses

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

 

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
Example 2:

Input: n = 1
Output: ["()"]
"""


class Solution:
    def generateParenthesis(self, n):
        res = []
        recur_generate_parenthesis(n, [], 0, 0, res)

        return res


def recur_generate_parenthesis(n, curr, left, right, res):
    if len(curr) == 2 * n:
        res.append(''.join(curr))
        return

    if left < n:
        recur_generate_parenthesis(n, curr + ['('], left + 1, right, res)
    if right < left:
        recur_generate_parenthesis(n, curr + [')'], left, right + 1, res)


# -----------------------------------------------------------------------
"""
78. Subsets

Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

 

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
Example 2:

Input: nums = [0]
Output: [[],[0]]
"""


class Solution:
    def subsets(self, nums):

        if not nums:
            return [[]]

        res = [[]]
        for num in nums:
            curr_list = []
            for list_ in res:
                curr_list.append(list(list_) + [num])
            res.extend(curr_list)

        return res

    # time O(2^n * n)
    # space O(2^n)


# -----------------------------------------------------------------------
"""
739. Daily Temperatures

Given a list of daily temperatures T, return a list such that, for each day in the input, 
tells you how many days you would have to wait until a warmer temperature. 
If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], 
your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will 
be an integer in the range [30, 100].
"""


class Solution:
    def dailyTemperatures(self, T):

        next_warmer = [float('inf') for _ in range(101)]

        for idx in range(len(T) - 1, -1, -1):
            warmer_index = float('inf')
            for curr_val in range(T[idx] + 1, 101):
                warmer_index = min(warmer_index, next_warmer[curr_val])

            next_warmer[T[idx]] = idx
            if warmer_index < float('inf'):
                T[idx] = warmer_index - idx
            else:
                T[idx] = 0

        return T

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
347. Top K Frequent Elements

Given a non-empty array of integers, return the k most frequent elements.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
Note:

You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
It's guaranteed that the answer is unique, in other words the set of the top k frequent elements is unique.
You can return the answer in any order.
"""

from collections import Counter
import heapq


class Solution:
    def topKFrequent(self, nums, k):
        min_heap = []
        count = Counter(nums)

        for num in count:
            if len(min_heap) == k:
                curr = heapq.heappop(min_heap)
                if curr[0] < count[num]:
                    curr = (count[num], num)
                heapq.heappush(min_heap, curr)
            else:
                heapq.heappush(min_heap, (count[num], num))

        return [val[1] for val in min_heap]

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
230. Kth Smallest Element in a BST

Given the root of a binary search tree, and an integer k, return the kth (1-indexed) smallest element in the tree.

 

Example 1:

Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
 

Constraints:

The number of nodes in the tree is n.
1 <= k <= n <= 104
0 <= Node.val <= 104
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root, k):
        stack = []

        while True:
            while root:
                stack.append(root)
                root = root.left

            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right

        # time O(h)
        # space O(h)


# -----------------------------------------------------------------------
"""
3. Longest Substring Without Repeating Characters

Given a string s, find the length of the longest substring without repeating characters.

 

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left, res, chars_map = 0, 0, {}
        n = len(s)

        for i in range(n):
            if s[i] in chars_map:
                prev_idx = chars_map[s[i]]
                if prev_idx >= left:
                    left = prev_idx + 1

            res = max(res, i - left + 1)
            chars_map[s[i]] = i

        return res

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
11. Container With Most Water

Given n non-negative integers a1, a2, ..., an , where each represents a point at
coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). 
Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

Example:

Input: [1,8,6,2,5,4,8,3,7]
Output: 49

"""


class Solution:
    def maxArea(self, height):
        left, right = 0, len(height) - 1
        max_area = 0
        while left < right:
            curr_area = (right - left) * (min(height[left], height[right]))
            max_area = max(max_area, curr_area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
15. 3Sum

Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []
Example 3:

Input: nums = [0]
Output: []
"""


class Solution:
    def threeSum(self, nums):
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if nums[i] > 0:
                break
            res += get_all_triples(nums, nums[i], i + 1, n - 1)

        return res

    # time O(n^2)
    # space O(n)


def get_all_triples(nums, curr_num, start, end):
    res = []
    while start < end:
        curr_sum = curr_num + nums[start] + nums[end]
        if curr_sum > 0:
            end -= 1
        elif curr_sum < 0:
            start += 1
        else:
            res.append([curr_num, nums[start], nums[end]])
            start += 1
            end -= 1
            while start < end and nums[start] == nums[start - 1]:
                start += 1
    return res


# -----------------------------------------------------------------------
"""
31. Next Permutation

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).

The replacement must be in place and use only constant extra memory.

 

Example 1:

Input: nums = [1,2,3]
Output: [1,3,2]
Example 2:

Input: nums = [3,2,1]
Output: [1,2,3]
Example 3:

Input: nums = [1,1,5]
Output: [1,5,1]
Example 4:

Input: nums = [1]
Output: [1]
"""


class Solution:
    def nextPermutation(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """

        i = len(nums) - 2

        while i >= 0 and nums[i + 1] <= nums[i]:
            i -= 1

        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1

            nums[i], nums[j] = nums[j], nums[i]

        if i == -1:
            nums.reverse()
        else:
            start, end = i + 1, len(nums) - 1
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                end -= 1
                start += 1

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
1277. Count Square Submatrices with All Ones

Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.

 

Example 1:

Input: matrix =
[
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]
Output: 15
Explanation: 
There are 10 squares of side 1.
There are 4 squares of side 2.
There is  1 square of side 3.
Total number of squares = 10 + 4 + 1 = 15.
Example 2:

Input: matrix = 
[
  [1,0,1],
  [1,1,0],
  [1,1,0]
]
Output: 7
Explanation: 
There are 6 squares of side 1.  
There is 1 square of side 2. 
Total number of squares = 6 + 1 = 7.
"""


class Solution:
    def countSquares(self, matrix):
        res = 0
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == 1:
                    res += 1
                if row == 0 or col == 0 or matrix[row][col] == 0:
                    continue
                min_val = get_min_val(matrix, row, col)
                matrix[row][col] += min_val
                res += min_val

        return res

    # time O(n * m)
    # space O(1)


def get_min_val(mat, row, col):
    up = mat[row - 1][col]
    left = mat[row][col - 1]
    diagonal = mat[row - 1][col - 1]
    return min(up, left, diagonal)


# -----------------------------------------------------------------------
"""
1110. Delete Nodes And Return Forest

Given the root of a binary tree, each node in the tree has a distinct value.

After deleting all nodes with a value in to_delete, we are left with a forest (a disjoint union of trees).

Return the roots of the trees in the remaining forest.  You may return the result in any order.
 

Example 1:

Input: root = [1,2,3,4,5,6,7], to_delete = [3,5]
Output: [[1,2,null,4],[6],[7]]
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def delNodes(self, root, to_delete):
        to_delete = set(to_delete)

        forests = []
        delete_vals(root, to_delete, forests)
        if root.val not in to_delete:
            forests.append(root)

        return forests

    # time O(n)
    # space O(n)


def delete_vals(head, vals_to_delete, forests):
    if not head:
        return None

    head.left = delete_vals(head.left, vals_to_delete, forests)
    head.right = delete_vals(head.right, vals_to_delete, forests)

    if head.val in vals_to_delete:
        if head.left:
            forests.append(head.left)
        if head.right:
            forests.append(head.right)

        return None

    return head


# -----------------------------------------------------------------------
"""
43. Multiply Strings

Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.

 

Example 1:

Input: num1 = "2", num2 = "3"
Output: "6"
Example 2:

Input: num1 = "123", num2 = "456"
Output: "56088"
 

Constraints:

1 <= num1.length, num2.length <= 200
num1 and num2 consist of digits only.
Both num1 and num2 do not contain any leading zero, except the number 0 itself.
"""


class Solution:
    def multiply(self, num1: str, num2: str) -> str:

        if num1 == '0' or num2 == '0':
            return '0'

        if len(num2) < len(num1):
            return self.multiply(num2, num1)

        num1 = [int(dig) for dig in reversed(num1)]
        num2 = [int(dig) for dig in reversed(num2)]
        res = []
        for i in range(len(num1)):
            curr_mult = [0] * i

            curr_mult += mult_digit(num1[i], num2)
            res = sum_nums(res, curr_mult)

        return ''.join([str(dig) for dig in reversed(res)])

    # time O(n * m)
    # space O(n + m)


def sum_nums(num1, num2):
    res = []
    carry = 0

    p1, p2 = 0, 0

    while p1 < len(num1) and p2 < len(num2):
        sum_ = num1[p1] + num2[p2] + carry
        res.append(sum_ % 10)
        carry = sum_ // 10
        p1 += 1
        p2 += 1

    while p1 < len(num1):
        sum_ = num1[p1] + carry
        res.append(sum_ % 10)
        carry = sum_ // 10
        p1 += 1

    while p2 < len(num2):
        sum_ = num2[p2] + carry
        res.append(sum_ % 10)
        carry = sum_ // 10
        p2 += 1

    if carry:
        res.append(1)

    return res


def mult_digit(digit, num):
    res = []
    carry = 0
    for dig in num:
        mult = (digit * dig) + carry
        res.append(mult % 10)
        carry = mult // 10

    if carry:
        res.append(carry)

    return res


# -----------------------------------------------------------------------
"""
48. Rotate Image

You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:

Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

Example 3:

Input: matrix = [[1]]
Output: [[1]]

Example 4:

Input: matrix = [[1,2],[3,4]]
Output: [[3,1],[4,2]]
"""


class Solution:
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        N = len(matrix)
        for row in range(N // 2):
            for col in range(row, N - row - 1):
                rotate(matrix, row, col)

        # time O(n^2)
        # space O(1)


def rotate(matrix, curr_row, curr_col):
    N = len(matrix)
    top_left = matrix[curr_row][curr_col]
    matrix[curr_row][curr_col] = matrix[N - curr_col - 1][curr_row]
    matrix[N - curr_col - 1][curr_row] = matrix[N - curr_row - 1][N - curr_col - 1]
    matrix[N - curr_row - 1][N - curr_col - 1] = matrix[curr_col][N - curr_row - 1]
    matrix[curr_col][N - curr_row - 1] = top_left


# -----------------------------------------------------------------------

"""
55. Jump Game

Given an array of non-negative integers nums, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

 

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
"""


class Solution:
    def canJump(self, nums):
        n = len(nums)
        lastIdx = n - 1

        for i in range(n - 2, -1, -1):
            if i + nums[i] >= lastIdx:
                lastIdx = i

        return lastIdx == 0

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
951. Flip Equivalent Binary Trees
Medium

918

47

Add to List

Share
For a binary tree T, we can define a flip operation as follows: choose any node, and swap the left and right child subtrees.

A binary tree X is flip equivalent to a binary tree Y if and only if we can make X equal to Y after some number of flip operations.

Given the roots of two binary trees root1 and root2, return true if the two trees are flip equivelent or false otherwise.

 

Example 1:

Flipped Trees Diagram
Input: root1 = [1,2,3,4,5,6,null,null,null,7,8], root2 = [1,3,2,null,6,4,5,null,null,null,null,8,7]
Output: true
Explanation: We flipped at nodes with values 1, 3, and 5.
Example 2:

Input: root1 = [], root2 = []
Output: true
Example 3:

Input: root1 = [], root2 = [1]
Output: false
Example 4:

Input: root1 = [0,null,1], root2 = []
Output: false
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def flipEquiv(self, root1, root2):

        if not root1 and not root2:
            return True

        if not root1 or not root2 or root1.val != root2.val:
            return False

        return (self.flipEquiv(root1.left, root2.left) or self.flipEquiv(root1.left, root2.right)) and (
                self.flipEquiv(root1.right, root2.left) or self.flipEquiv(root1.right, root2.right))

    # time O(n + m)
    # space O(h)


# -----------------------------------------------------------------------
"""
1219. Path with Maximum Gold

In a gold mine grid of size m * n, each cell in this mine has an integer representing the amount of gold in that cell, 0 if it is empty.

Return the maximum amount of gold you can collect under the conditions:

Every time you are located in a cell you will collect all the gold in that cell.
From your position you can walk one step to the left, right, up or down.
You can't visit the same cell more than once.
Never visit a cell with 0 gold.
You can start and stop collecting gold from any position in the grid that has some gold.
 

Example 1:

Input: grid = [[0,6,0],[5,8,7],[0,9,0]]
Output: 24
Explanation:
[[0,6,0],
 [5,8,7],
 [0,9,0]]
Path to get the maximum gold, 9 -> 8 -> 7.
Example 2:

Input: grid = [[1,0,7],[2,0,6],[3,4,5],[0,3,0],[9,0,20]]
Output: 28
Explanation:
[[1,0,7],
 [2,0,6],
 [3,4,5],
 [0,3,0],
 [9,0,20]]
Path to get the maximum gold, 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7.
 

Constraints:

1 <= grid.length, grid[i].length <= 15
0 <= grid[i][j] <= 100
There are at most 25 cells containing gold.
"""


class Solution:
    def getMaximumGold(self, grid):

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                dfs_visit(grid, row, col)

        global max_gold
        return max_gold

    # time O(n * m)
    # space O(1) since the longets path is 25


def is_valid(grid, row, col):
    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or grid[row][col] == 0:
        return False

    return True


max_gold = 0
curr_gold = 0


def dfs_visit(grid, row, col):
    if not is_valid(grid, row, col):
        return

    global curr_gold
    cell_gold = grid[row][col]
    curr_gold += cell_gold
    grid[row][col] = 0

    global max_gold
    max_gold = max(max_gold, curr_gold)

    for n_row, n_col in get_neighbors(row, col):
        dfs_visit(grid, n_row, n_col)

    curr_gold -= cell_gold
    grid[row][col] = cell_gold


directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def get_neighbors(row, col):
    global directions
    neighbors = []

    for dx, dy in directions:
        neighbors.append((row + dx, col + dy))

    return neighbors


# -----------------------------------------------------------------------
"""
159. Longest Substring with At Most Two Distinct Characters

Given a string s , find the length of the longest substring t  that contains at most 2 distinct characters.

Example 1:

Input: "eceba"
Output: 3
Explanation: t is "ece" which its length is 3.
Example 2:

Input: "ccaabbb"
Output: 5
Explanation: t is "aabbb" which its length is 5.
"""

from collections import defaultdict


class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        counts = defaultdict(int)
        left = 0

        start_idx, length = 0, 0
        for i, char in enumerate(s):
            counts[char] += 1
            if len(counts) <= 2:
                if i - left + 1 > length:
                    length = i - left + 1
                    start_idx = left
            else:
                while left < i and len(counts) > 2:
                    counts[s[left]] -= 1
                    if counts[s[left]] == 0:
                        del counts[s[left]]
                    left += 1

        return length

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
695. Max Area of Island

Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)

Example 1:

[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
Given the above grid, return 6. Note the answer is not 11, because the island must be connected 4-directionally.
Example 2:

[[0,0,0,0,0,0,0,0]]
Given the above grid, return 0.
Note: The length of each dimension in the given grid does not exceed 50.
"""


class Solution:
    def maxAreaOfIsland(self, grid):

        max_area = 0
        visited = set()
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                max_area = max(max_area, get_island_area(grid, visited, row, col))

        return max_area

    # time O(n * m)
    # space O(n * m)


def is_valid(grid, row, col):
    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or grid[row][col] == 0:
        return False

    return True


def get_neighbors_directions():
    return [(-1, 0), (0, 1), (1, 0), (0, -1)]


def get_island_area(grid, visited, row, col):
    if not is_valid(grid, row, col) or (row, col) in visited:
        return 0

    visited.add((row, col))
    area = 1
    for dx, dy in get_neighbors_directions():
        area += get_island_area(grid, visited, row + dx, col + dy)

    return area


"""
0 1 1 0
0 1 0 0
1 0 1 0

visited = ((0, 1), (0, 2), (1, 1)), max_area = 3
row = 0, col = 1 => area = 1 + 1 + 1 => return 3
    row = 0, col = 2 => return 1
    row = 1, col = 1 => return 1
    
"""

# -----------------------------------------------------------------------
"""
946. Validate Stack Sequences

Given two sequences pushed and popped with distinct values, return true if and only if this could have been the result of a sequence of push and pop operations on an initially empty stack.

 

Example 1:

Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
Output: true
Explanation: We might do the following sequence:
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
Example 2:

Input: pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
Output: false
Explanation: 1 cannot be popped before 2.
 

Constraints:

0 <= pushed.length == popped.length <= 1000
0 <= pushed[i], popped[i] < 1000
pushed is a permutation of popped.
pushed and popped have distinct values.
"""


class Solution:
    def validateStackSequences(self, pushed, popped):

        if len(pushed) != len(popped):
            return False

        push_idx, pop_idx = 0, 0
        stack = []

        while push_idx < len(pushed):
            stack.append(pushed[push_idx])
            push_idx += 1

            while stack and pop_idx < len(popped) and stack[-1] == popped[pop_idx]:
                stack.pop()
                pop_idx += 1

        return len(stack) == 0

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
681. Next Closest Time

Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.

You may assume the given input string is always valid. For example, "01:34", "12:09" are all valid. "1:34", "12:9" are all invalid.



Example 1:

Input: time = "19:34"
Output: "19:39"
Explanation: The next closest time choosing from digits 1, 9, 3, 4, is 19:39, which occurs 5 minutes later.
It is not 19:33, because this occurs 23 hours and 59 minutes later.
Example 2:

Input: time = "23:59"
Output: "22:22"
Explanation: The next closest time choosing from digits 2, 3, 5, 9, is 22:22.
It may be assumed that the returned time is next day's time since it is smaller than the input time numerically.
"""

SECS = 60
HOURS = 24


class Solution:
    def nextClosestTime(self, time: str) -> str:
        hours, minutes = time.split(':')
        total_mins = int(hours) * SECS + int(minutes)

        valid_digits = set(get_integers(hours) + get_integers(minutes))
        while True:
            total_mins = (total_mins + 1) % (SECS * HOURS)
            curr_digits = set(get_digits(total_mins))

            if curr_digits.intersection(valid_digits) == curr_digits:
                return formate_time(total_mins)

    # time O(1)
    # space O(1)


def get_integers(strs):
    return list(map(lambda x: int(x), list(strs)))


def get_digits(minutes):
    res = []
    res.append(minutes // 60 // 10)  # left hours
    res.append(minutes // 60 % 10)  # right hours
    res.append(minutes % 60 // 10)  # left minutes
    res.append(minutes % 60 % 10)  # right minutes

    return res


def formate_time(minutes):
    digits = get_digits(minutes)

    return f'{digits[0]}{digits[1]}:{digits[2]}{digits[3]}'


# -----------------------------------------------------------------------
"""
19. Remove Nth Node From End of List

Given the head of a linked list, remove the nth node from the end of the list and return its head.

Follow up: Could you do this in one pass?

 

Example 1:


Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
Example 2:

Input: head = [1], n = 1
Output: []
Example 3:

Input: head = [1,2], n = 1
Output: [1]
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head, n):
        dummy = ListNode('DUMMY')
        dummy.next = head
        prev, curr = dummy, dummy

        while curr and n >= 0:
            curr = curr.next
            n -= 1

        while curr:
            curr = curr.next
            prev = prev.next

        prev.next = prev.next.next

        return dummy.next

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

Example 1:


Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]
"""


class Solution:
    def addTwoNumbers(self, l1, l2):
        p1, p2 = l1, l2
        res = curr = ListNode('DUMMY')
        curr_sum = 0
        carry = 0
        while p1 and p2:
            new_node = ListNode(0)
            curr_sum = p1.val + p2.val + carry
            new_node.val = curr_sum % 10
            curr.next = new_node
            carry = curr_sum // 10
            p1 = p1.next
            p2 = p2.next
            curr = curr.next

        while p2:
            new_node = ListNode(0)
            if carry == 0:
                curr.next = p2
                return res.next

            curr_sum = p2.val + carry
            new_node.val = curr_sum % 10
            curr.next = new_node
            carry = curr_sum // 10
            p2 = p2.next
            curr = curr.next

        while p1:
            new_node = ListNode(0)
            if carry == 0:
                curr.next = p1
                return res.next

            curr_sum = p1.val + carry
            new_node.val = curr_sum % 10
            curr.next = new_node
            carry = curr_sum // 10
            p1 = p1.next
            curr = curr.next

        if carry:
            curr.next = ListNode(carry)

        return res.next

    # time O(n + m)
    # space O(n + m)


# -----------------------------------------------------------------------
"""
138. Copy List with Random Pointer

A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.

 

Example 1:


Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
Example 2:


Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]
Example 3:



Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]
"""

"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""


class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        cloned_list = Node(0)

        curr = head
        cloned_curr = cloned_list

        # point to the old appropriate node by the random poitner
        while curr:
            new_node = Node(curr.val)
            cloned_curr.next = new_node
            next_ = curr.next

            curr.next = new_node
            new_node.random = curr

            curr = next_
            cloned_curr = cloned_curr.next

        curr = cloned_list.next

        # take the appropriate random from the new cloned list
        while curr:
            if curr.random.random:
                curr.random = curr.random.random.next
            else:
                curr.random = None
            curr = curr.next

        return cloned_list.next

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
1376. Time Needed to Inform All Employees

A company has n employees with a unique ID for each employee from 0 to n - 1. The head of the company is the one with headID.

Each employee has one direct manager given in the manager array where manager[i] is the direct manager of the i-th employee, manager[headID] = -1. Also, it is guaranteed that the subordination relationships have a tree structure.

The head of the company wants to inform all the company employees of an urgent piece of news. He will inform his direct subordinates, and they will inform their subordinates, and so on until all employees know about the urgent news.

The i-th employee needs informTime[i] minutes to inform all of his direct subordinates (i.e., After informTime[i] minutes, all his direct subordinates can start spreading the news).

Return the number of minutes needed to inform all the employees about the urgent news.

 

Example 1:

Input: n = 1, headID = 0, manager = [-1], informTime = [0]
Output: 0
Explanation: The head of the company is the only employee in the company.
Example 2:


Input: n = 6, headID = 2, manager = [2,2,-1,2,2,2], informTime = [0,0,1,0,0,0]
Output: 1
Explanation: The head of the company with id = 2 is the direct manager of all the employees in the company and needs 1 minute to inform them all.
The tree structure of the employees in the company is shown.
Example 3:


Input: n = 7, headID = 6, manager = [1,2,3,4,5,6,-1], informTime = [0,6,5,4,3,2,1]
Output: 21
Explanation: The head has id = 6. He will inform employee with id = 5 in 1 minute.
The employee with id = 5 will inform the employee with id = 4 in 2 minutes.
The employee with id = 4 will inform the employee with id = 3 in 3 minutes.
The employee with id = 3 will inform the employee with id = 2 in 4 minutes.
The employee with id = 2 will inform the employee with id = 1 in 5 minutes.
The employee with id = 1 will inform the employee with id = 0 in 6 minutes.
Needed time = 1 + 2 + 3 + 4 + 5 + 6 = 21.
Example 4:

Input: n = 15, headID = 0, manager = [-1,0,0,1,1,2,2,3,3,4,4,5,5,6,6], informTime = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
Output: 3
Explanation: The first minute the head will inform employees 1 and 2.
The second minute they will inform employees 3, 4, 5 and 6.
The third minute they will inform the rest of employees.
"""

from collections import deque, defaultdict


class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        total_time = 0
        graph = construct_graph(n, manager)

        q = deque([(headID, 0)])

        while len(q) > 0:
            curr_id, curr_time = q.popleft()
            total_time = max(total_time, curr_time)
            for adj in graph[curr_id]:
                q.append((adj, curr_time + informTime[curr_id]))

        return total_time

    # time O(n)
    # space O(n)


def construct_graph(n, manager):
    graph = defaultdict(list)
    for i in range(n):
        graph[i] = []

    for idx, val in enumerate(manager):
        if val == -1:
            continue
        graph[val].append(idx)

    return graph


# -----------------------------------------------------------------------
"""
809. Expressive Words

Sometimes people repeat letters to represent extra feeling, such as "hello" -> "heeellooo", "hi" -> "hiiii".  In these strings like "heeellooo", we have groups of adjacent letters that are all the same:  "h", "eee", "ll", "ooo".

For some given string S, a query word is stretchy if it can be made to be equal to S by any number of applications of the following extension operation: choose a group consisting of characters c, and add some number of characters c to the group so that the size of the group is 3 or more.

For example, starting with "hello", we could do an extension on the group "o" to get "hellooo", but we cannot get "helloo" since the group "oo" has size less than 3.  Also, we could do another extension like "ll" -> "lllll" to get "helllllooo".  If S = "helllllooo", then the query word "hello" would be stretchy because of these two extension operations: query = "hello" -> "hellooo" -> "helllllooo" = S.

Given a list of query words, return the number of words that are stretchy. 

 

Example:
Input: 
S = "heeellooo"
words = ["hello", "hi", "helo"]
Output: 1
Explanation: 
We can extend "e" and "o" in the word "hello" to get "heeellooo".
We can't extend "helo" to get "heeellooo" because the group "ll" is not size 3 or more.
"""


class Solution:
    def expressiveWords(self, S, words):

        res = 0
        s_chars_count = count_chars(S)
        for word in words:
            if can_form_s(s_chars_count, word):
                res += 1

        return res

    # time O(n * m)
    # space O(n * m)


def count_chars(word):
    res = []
    i = 0
    while i < len(word):
        curr_count = 0
        curr_char = word[i]
        while i < len(word) and word[i] == curr_char:
            curr_count += 1
            i += 1
        res += [curr_char, str(curr_count)]

    return ''.join(res)


def can_form_s(s_chars_count, word):
    word_chars_count = count_chars(word)
    if len(s_chars_count) != len(word_chars_count):
        return False

    for i in range(0, len(word_chars_count), 2):
        curr_char, curr_count = word_chars_count[i], int(word_chars_count[i + 1])
        s_char, s_count = s_chars_count[i], int(s_chars_count[i + 1])

        if curr_char != s_char:
            return False
        elif curr_count == s_count:
            continue
        elif s_count > curr_count + 1:
            continue
        elif s_count - curr_count == 1 and s_count > 2:
            continue
        else:
            return False

    return True


# -----------------------------------------------------------------------
"""
833. Find And Replace in String

To some string S, we will perform some replacement operations that replace groups of letters with new ones (not necessarily the same size).

Each replacement operation has 3 parameters: a starting index i, a source word x and a target word y.  The rule is that if x starts at position i in the original string S, then we will replace that occurrence of x with y.  If not, we do nothing.

For example, if we have S = "abcd" and we have some replacement operation i = 2, x = "cd", y = "ffff", then because "cd" starts at position 2 in the original string S, we will replace it with "ffff".

Using another example on S = "abcd", if we have both the replacement operation i = 0, x = "ab", y = "eee", as well as another replacement operation i = 2, x = "ec", y = "ffff", this second operation does nothing because in the original string S[2] = 'c', which doesn't match x[0] = 'e'.

All these operations occur simultaneously.  It's guaranteed that there won't be any overlap in replacement: for example, S = "abc", indexes = [0, 1], sources = ["ab","bc"] is not a valid test case.

 

Example 1:

Input: S = "abcd", indexes = [0, 2], sources = ["a", "cd"], targets = ["eee", "ffff"]
Output: "eeebffff"
Explanation:
"a" starts at index 0 in S, so it's replaced by "eee".
"cd" starts at index 2 in S, so it's replaced by "ffff".
Example 2:

Input: S = "abcd", indexes = [0, 2], sources = ["ab","ec"], targets = ["eee","ffff"]
Output: "eeecd"
Explanation:
"ab" starts at index 0 in S, so it's replaced by "eee".
"ec" doesn't starts at index 2 in the original S, so we do nothing.
"""


class Solution:
    def findReplaceString(self, S, indexes, sources, targets):
        res = []
        operations = get_operations(indexes, sources, targets)
        j = 0
        i = 0
        while i < len(S) and j < len(operations):
            if i != operations[j][0]:
                res.append(S[i])
                i += 1
                continue
            elif i == operations[j][0]:
                if can_replace(S, i, j, operations):
                    res.append(operations[j][2])
                else:
                    res.append(S[i:i + len(operations[j][1])])

                i += len(operations[j][1])
                j += 1

        if i < len(S):
            res.append(S[i:])

        return ''.join(res)

    # time O(n * m)
    # space O(n)


def get_operations(indexes, sources, targets):
    res = []
    for i in range(len(indexes)):
        res.append((indexes[i], sources[i], targets[i]))

    return sorted(res, key=lambda x: x[0])


def can_replace(str_, i, j, operations):
    length = len(operations[j][1])

    if str_[i: i + length] == operations[j][1]:
        return True

    return False


# -----------------------------------------------------------------------
"""
849. Maximize Distance to Closest Person

You are given an array representing a row of seats where seats[i] = 1 represents a person sitting in the ith seat, and seats[i] = 0 represents that the ith seat is empty (0-indexed).

There is at least one empty seat, and at least one person sitting.

Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. 

Return that maximum distance to the closest person.

 

Example 1:


Input: seats = [1,0,0,0,1,0,1]
Output: 2
Explanation: 
If Alex sits in the second open seat (i.e. seats[2]), then the closest person has distance 2.
If Alex sits in any other open seat, the closest person has distance 1.
Thus, the maximum distance to the closest person is 2.
Example 2:

Input: seats = [1,0,0,0]
Output: 3
Explanation: 
If Alex sits in the last seat (i.e. seats[3]), the closest person is 3 seats away.
This is the maximum distance possible, so the answer is 3.
Example 3:

Input: seats = [0,1]
Output: 1
"""


class Solution:
    def maxDistToClosest(self, seats):
        if not seats:
            return -1

        left, right = -1, -1
        n = len(seats)
        max_dist = 0
        i = 0

        while i < n:

            while i < n and seats[i] == 1:
                i += 1
            left = i

            while i < n and seats[i] == 0:
                i += 1
            right = i

            if left == 0 or right == n:
                max_dist = max(max_dist, right - left)
            else:
                curr_dist = (right - left + 1) // 2
                max_dist = max(max_dist, curr_dist)

        return max_dist

    # time O(n)
    # space O(1)


#         n = len(seats)
#         left_dist = [n for _ in range(n)]
#         right_dist = left_dist.copy()


#         for i in range(n):
#             if seats[i] == 1:
#                 left_dist[i] = 0
#             elif i > 0:
#                 left_dist[i] = left_dist[i - 1] + 1

#         for i in range(n - 1, -1, -1):
#             if seats[i] == 1:
#                 right_dist[i] = 0
#             elif i < n - 1:
#                 right_dist[i] = right_dist[i + 1] + 1

#         return max(min(left_dist[i], right_dist[i]) for i in range(n))

#     # time O(n)
#     # space O(n)

# -----------------------------------------------------------------------

"""
1314. Matrix Block Sum

Given a m * n matrix mat and an integer K, return a matrix answer where each answer[i][j] is the sum of all elements mat[r][c] for i - K <= r <= i + K, j - K <= c <= j + K, and (r, c) is a valid position in the matrix.
 

Example 1:

Input: mat = [[1,2,3],[4,5,6],[7,8,9]], K = 1
Output: [[12,21,16],[27,45,33],[24,39,28]]
Example 2:

Input: mat = [[1,2,3],[4,5,6],[7,8,9]], K = 2
Output: [[45,45,45],[45,45,45],[45,45,45]]
"""


class Solution:
    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        n = len(mat)
        m = len(mat[0])

        pre_sum = get_pre_sum(mat, n, m)

        res = [[0 for _ in range(m)] for _ in range(n)]

        for row in range(1, n + 1):
            for col in range(1, m + 1):
                starti, startj = max(1, row - K), max(1, col - K)
                endi, endj = min(n, row + K), min(m, col + K)

                bottom_right = pre_sum[endi][endj]
                top_right = pre_sum[starti - 1][endj]
                bottom_left = pre_sum[endi][startj - 1]
                top_left = pre_sum[starti - 1][startj - 1]

                res[row - 1][col - 1] = bottom_right - top_right - bottom_left + top_left

        return res

    # time O(n *m)
    # space O(n * m)


def get_pre_sum(mat, n, m):
    pre_sum = [[0 for col in range(m + 1)] for row in range(n + 1)]

    for row in range(1, n + 1):
        for col in range(1, m + 1):
            curr_val = mat[row - 1][col - 1]
            top = pre_sum[row - 1][col]
            left = pre_sum[row][col - 1]
            diagonal = pre_sum[row - 1][col - 1]
            pre_sum[row][col] = curr_val + top + left - diagonal

    return pre_sum


# -----------------------------------------------------------------------
"""
846. Hand of Straights

Alice has a hand of cards, given as an array of integers.

Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.

Return true if and only if she can.

Note: This question is the same as 1296: https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/

 

Example 1:

Input: hand = [1,2,3,6,2,3,4,7,8], W = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]
"""


class Solution:
    def isNStraightHand(self, hand: List[int], W: int) -> bool:
        if len(hand) % W != 0:
            return False

        hand = Counter(hand)
        curr_num = min(hand.keys())
        curr_length = 0

        while len(hand) > 0:
            if curr_num in hand:
                hand[curr_num] -= 1
                if hand[curr_num] == 0:
                    del hand[curr_num]
                curr_length += 1
                curr_num += 1
            else:
                if curr_length == 0:
                    if hand:
                        curr_num = min(hand.keys())
                else:
                    return False

            if curr_length == W:
                curr_length = 0
                if hand:
                    curr_num = min(hand.keys())

        return True

    # time O(n * (n / w))
    # space O(n)


# -----------------------------------------------------------------------
"""
729. My Calendar I

Implement a MyCalendar class to store your events. A new event can be added if adding the event will not cause a double booking.

Your class will have the method, book(int start, int end). Formally, this represents a booking on the half open interval [start, end), the range of real numbers x such that start <= x < end.

A double booking happens when two events have some non-empty intersection (ie., there is some time that is common to both events.)

For each call to the method MyCalendar.book, return true if the event can be added to the calendar successfully without causing a double booking. Otherwise, return false and do not add the event to the calendar.

Your class will be called like this: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)
Example 1:

MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(15, 25); // returns false
MyCalendar.book(20, 30); // returns true
Explanation: 
The first event can be booked.  The second can't because time 15 is already booked by another event.
The third event can be booked, as the first event takes every time less than 20, but not including 20.
"""


class Node:
    __slots__ = 'start', 'end', 'left', 'right'

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left = self.right = None

    def insert(self, node):
        if node.start >= self.end:
            if not self.right:
                self.right = node
                return True
            return self.right.insert(node)
        elif node.end <= self.start:
            if not self.left:
                self.left = node
                return True
            return self.left.insert(node)
        else:
            return False


class MyCalendar(object):
    def __init__(self):
        self.root = None

    def book(self, start, end):
        if self.root is None:
            self.root = Node(start, end)
            return True
        return self.root.insert(Node(start, end))

    # time O(n^2) / O(n*log(n))
    # space O(n)


# -----------------------------------------------------------------------

"""
215. Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:

Input: [3,2,1,5,6,4] and k = 2
Output: 5
Example 2:

Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
"""

import heapq


class Solution:
    def findKthLargest(self, nums, k):

        k_largest = []
        for num in nums:
            heapq.heappush(k_largest, num)
            if len(k_largest) > k:
                heapq.heappop(k_largest)

        return heapq.heappop(k_largest)

    # time O(n * log(k))
    # space O(k)

# -----------------------------------------------------------------------
"""
253. Meeting Rooms II

Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.

 

Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
Example 2:

Input: intervals = [[7,10],[2,4]]
Output: 1
"""


class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """

        events = []
        for time in intervals:
            events.append((time[0], 'b'))
            events.append((time[1], 'a'))

        events.sort()

        count = 0
        rooms_num = 0
        for time, event in events:
            if event == 'a':
                count -= 1
            else:
                count += 1

            rooms_num = max(rooms_num, count)

        return rooms_num

    # time O(n * log(n))
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
