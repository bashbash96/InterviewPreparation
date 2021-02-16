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
