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


class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n < 2:
            return n

        max_len = 1

        memo = [0] * n
        memo[0] = nums[0]

        for i in range(1, n):
            idx = bin_search(memo, 0, max_len - 1, nums[i])

            memo[idx] = nums[i]
            if idx == max_len:
                max_len += 1

        return max_len

    # time O(n * log(n))
    # space O(n)


def bin_search(arr, start, end, num):
    while start <= end:
        mid = (start + end) >> 1

        if num < arr[mid]:
            end = mid - 1
        elif num > arr[mid]:
            start = mid + 1
        else:
            return mid

    return start


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
    def __init__(self, val, key):
        self.val = val
        self.key = key
        self.next = None
        self.prev = None


class DoubleLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, value, key):
        node = Node(value, key)

        if not self.head:
            self.head = self.tail = node
            return node

        node.next = self.head
        self.head.prev = node
        self.head = node

        return node

    def delete(self, node):

        if node == self.head:
            self.head = node.next
            if not self.head:
                self.tail = None
            else:
                self.head.prev = None
            return

        if node == self.tail:
            self.tail = node.prev
            if not self.tail:
                self.head = None
            else:
                self.tail.next = None
            return

        node.prev.next = node.next
        node.next.prev = node.prev


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.list = DoubleLinkedList()
        self.capacity = capacity
        self.size = 0
        self.keys = {}

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """

        if key not in self.keys:
            return -1

        val = self.keys[key].val

        self.put(key, val)

        return val

    # time O(1)
    # space O(1)

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """

        if key in self.keys:
            self.list.delete(self.keys[key])
            del self.keys[key]
        else:
            self.size += 1

        if self.size > self.capacity:
            self.size = self.capacity
            del_key = self.list.tail.key
            self.list.delete(self.keys[del_key])
            del self.keys[del_key]

        node = self.list.add(value, key)
        self.keys[key] = node

    # time O(1)
    # space O(1)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

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
340. Longest Substring with At Most K Distinct Characters

Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters.

 

Example 1:

Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.
Example 2:

Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.
"""


class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """

        left = 0
        count = defaultdict(int)
        max_length = 0
        for i in range(len(s)):
            count[s[i]] += 1
            while len(count) > k:
                count[s[left]] -= 1
                if count[s[left]] == 0:
                    del count[s[left]]
                left += 1

            max_length = max(max_length, i - left + 1)

        return max_length

    # time O(n)
    # space O(k)


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
"""
652. Find Duplicate Subtrees

Given the root of a binary tree, return all duplicate subtrees.

For each kind of duplicate subtrees, you only need to return the root node of any one of them.

Two trees are duplicate if they have the same structure with the same node values.

 

Example 1:


Input: root = [1,2,3,4,null,2,4,null,null,4]
Output: [[2,4],[4]]
Example 2:


Input: root = [2,1,1]
Output: [[1]]
Example 3:


Input: root = [2,2,2,3,null,3,null]
Output: [[2,3],[3]]
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import defaultdict


class Solution(object):
    def findDuplicateSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: List[TreeNode]
        """
        count = defaultdict(int)
        res = []

        count_duplicates(root, res, count)

        return res

    # time O(n * h)
    # space O(n^2)


def count_duplicates(root, res, count):
    if not root:
        return '#'

    left = count_duplicates(root.left, res, count)
    right = count_duplicates(root.right, res, count)
    curr_path = "{},{},{}".format(root.val, left, right)

    count[curr_path] += 1
    if count[curr_path] == 2:
        res.append(root)

    return curr_path


# -----------------------------------------------------------------------
"""
1048. Longest String Chain

Given a list of words, each word consists of English lowercase letters.

Let's say word1 is a predecessor of word2 if and only if we can add exactly one letter anywhere in word1 to make it equal to word2.  For example, "abc" is a predecessor of "abac".

A word chain is a sequence of words [word_1, word_2, ..., word_k] with k >= 1, where word_1 is a predecessor of word_2, word_2 is a predecessor of word_3, and so on.

Return the longest possible length of a word chain with words chosen from the given list of words.

 

Example 1:

Input: words = ["a","b","ba","bca","bda","bdca"]
Output: 4
Explanation: One of the longest word chain is "a","ba","bda","bdca".
Example 2:

Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
Output: 5

"""

from collections import defaultdict


class Solution(object):
    def longestStrChain(self, words):
        """
        :type words: List[str]
        :rtype: int
        """

        str_idxs = get_indexes(words)
        graph = generate_graph(words, str_idxs)

        longest_word = 0
        memo = [0 for _ in range(len(words))]

        for i in range(len(words)):
            longest_word = max(longest_word, get_longest(graph, memo, i))

        return longest_word

    # time O(n * l^2)
    # space O(n)


def get_indexes(words):
    idxs = {}
    for idx, word in enumerate(words):
        idxs[word] = idx

    return idxs


def generate_graph(words, str_idxs):
    graph = defaultdict(set)

    for i, word in enumerate(words):
        for pos in range(len(word)):
            new_word = (word[:pos] + word[pos + 1:])
            if new_word in str_idxs:
                graph[str_idxs[new_word]].add(i)

    return graph


def get_longest(graph, memo, i):
    if memo[i] > 0:
        return memo[i]

    memo[i] = 1
    for adj in graph[i]:
        memo[i] = max(memo[i], get_longest(graph, memo, adj) + 1)

    return memo[i]


# -----------------------------------------------------------------------
"""
973. K Closest Points to Origin

We have a list of points on the plane.  Find the K closest points to the origin (0, 0).

(Here, the distance between two points on a plane is the Euclidean distance.)

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)

 

Example 1:

Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]
Explanation: 
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].
Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], K = 2
Output: [[3,3],[-2,4]]
(The answer [[-2,4],[3,3]] would also be accepted.)
"""

import heapq


class Solution(object):
    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """

        k_closest = []
        for point in points:
            dist = get_distance_from_origin(point)
            heapq.heappush(k_closest, (-dist, point))
            if len(k_closest) > k:
                heapq.heappop(k_closest)

        return [pair[1] for pair in k_closest]

    # time O(n * log(k))
    # space O(k)


def get_distance_from_origin(point):
    return pow(point[0], 2) + pow(point[1], 2)


# -----------------------------------------------------------------------
"""
1769. Minimum Number of Operations to Move All Balls to Each Box

You have n boxes. You are given a binary string boxes of length n, where boxes[i] is '0' if the ith box is empty, and '1' if it contains one ball.

In one operation, you can move one ball from a box to an adjacent box. Box i is adjacent to box j if abs(i - j) == 1. Note that after doing so, there may be more than one ball in some boxes.

Return an array answer of size n, where answer[i] is the minimum number of operations needed to move all the balls to the ith box.

Each answer[i] is calculated considering the initial state of the boxes.

 

Example 1:

Input: boxes = "110"
Output: [1,1,3]
Explanation: The answer for each box is as follows:
1) First box: you will have to move one ball from the second box to the first box in one operation.
2) Second box: you will have to move one ball from the first box to the second box in one operation.
3) Third box: you will have to move one ball from the first box to the third box in two operations, and move one ball from the second box to the third box in one operation.
Example 2:

Input: boxes = "001011"
Output: [11,8,5,4,3,4]
"""


class Solution(object):
    def minOperations(self, boxes):
        """
        :type boxes: str
        :rtype: List[int]
        """
        n = len(boxes)
        left_ones_count = []
        curr_count = 0
        for i, val in enumerate(boxes):
            left_ones_count.append(curr_count)
            if val == '1':
                curr_count += 1

        right_ones_count = []
        curr_count = 0
        for i in range(n - 1, -1, -1):
            right_ones_count.append(curr_count)
            if boxes[i] == '1':
                curr_count += 1

        right_ones_count = right_ones_count[::-1]

        return calculate_distances(boxes, left_ones_count, right_ones_count)

    # time O(n)
    # space O(n)


def calculate_distances(boxes, left_ones_count, right_ones_count):
    result = [0 for _ in range(len(boxes))]
    result[0] = get_distances(boxes)

    for i in range(1, len(boxes)):
        curr_res = result[i - 1]
        if boxes[i] == '1':
            curr_res -= 1
        curr_res += left_ones_count[i]
        curr_res -= right_ones_count[i]

        result[i] = curr_res

    return result


def get_distances(boxes):
    res = 0
    for i in range(1, len(boxes)):
        if boxes[i] == '1':
            res += i

    return res


# -----------------------------------------------------------------------
"""
721. Accounts Merge

Given a list accounts, each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some email that is common to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.

Example 1:
Input: 
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
Explanation: 
The first and third John's are the same person as they have the common email "johnsmith@mail.com".
The second John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
"""

from collections import defaultdict


class Solution(object):
    def accountsMerge(self, accounts):
        """
        :type accounts: List[List[str]]
        :rtype: List[List[str]]
        """
        if not accounts:
            return accounts

        email_to_name, graph = generate_graph(accounts)

        visited = set()
        res = []
        for email in graph:
            if email not in visited:
                curr_res = [email_to_name[email]] + sorted(get_component(graph, visited, email))
                res += [curr_res]

        return res

    # time O(n * m * log(m)) - n num of accounts, m - max emails number
    # space O(n * m)


def get_component(graph, visited, vertex):
    if vertex in visited:
        return []

    visited.add(vertex)

    curr = [vertex]
    for adj in graph[vertex]:
        curr += get_component(graph, visited, adj)

    return curr


def generate_graph(accounts):
    email_to_name = {}
    graph = defaultdict(set)

    for account in accounts:
        name = account[0]
        for email in account[1:]:
            graph[email].add(account[1])
            graph[account[1]].add(email)
            email_to_name[email] = name

    return email_to_name, graph


# -----------------------------------------------------------------------
"""
1138. Alphabet Board Path

On an alphabet board, we start at position (0, 0), corresponding to character board[0][0].

Here, board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"], as shown in the diagram below.



We may make the following moves:

'U' moves our position up one row, if the position exists on the board;
'D' moves our position down one row, if the position exists on the board;
'L' moves our position left one column, if the position exists on the board;
'R' moves our position right one column, if the position exists on the board;
'!' adds the character board[r][c] at our current position (r, c) to the answer.
(Here, the only positions that exist on the board are positions with letters on them.)

Return a sequence of moves that makes our answer equal to target in the minimum number of moves.  You may return any path that does so.

 

Example 1:

Input: target = "leet"
Output: "DDR!UURRR!!DDD!"
Example 2:

Input: target = "code"
Output: "RR!DDRR!UUL!R!"
"""

from collections import defaultdict, deque

board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]

directions = {'U': [-1, 0], 'R': [0, 1], 'D': [1, 0], 'L': [0, -1]}
add = '!'
ROWS = 6
COLS = 5


class Solution(object):
    def alphabetBoardPath(self, target):
        """
        :type target: str
        :rtype: str
        """

        path = []
        curr_pos = (0, 0)
        for char in target:
            curr_pos, curr_path = get_path(curr_pos, char)
            path += curr_path

        return ''.join(path)

    # time O(m * 26)
    # space O(m)


def get_path(curr_pos, char):
    row, col = curr_pos
    if board[row][col] == char:
        return curr_pos, [add]

    parent = defaultdict()
    visited = set()
    final_pos = ()

    q = deque([curr_pos])
    visited.add(curr_pos)

    while len(q) > 0:
        curr = q.popleft()

        row, col = curr
        if board[row][col] == char:
            final_pos = (row, col)
            break

        for direction in directions:
            dx, dy = directions[direction]
            new_row, new_col = row + dx, col + dy
            if is_valid(new_row, new_col) and not (new_row, new_col) in visited:
                parent[(new_row, new_col)] = direction
                visited.add((new_row, new_col))
                q.append((new_row, new_col))

    curr_path = generate_path(parent, final_pos, curr_pos)

    return final_pos, curr_path


def is_valid(row, col):
    if row < 0 or col < 0 or col >= COLS or row >= ROWS:
        return False

    if row == ROWS - 1 and col != 0:
        return False

    return True


def generate_path(parent, end_pos, start_pos):
    path = [add]

    curr_pos = end_pos

    while curr_pos != start_pos:
        path.append(parent[curr_pos])
        curr_pos = get_opposite_direction(curr_pos, parent[curr_pos])

    return path[::-1]


def get_opposite_direction(curr_pos, parent):
    row, col = curr_pos
    if parent == 'U':
        dx, dy = directions['D']
    elif parent == 'D':
        dx, dy = directions['U']
    elif parent == 'L':
        dx, dy = directions['R']
    else:
        dx, dy = directions['L']

    return row + dx, col + dy


# -----------------------------------------------------------------------
"""
200. Number of Islands

Given an m x n 2d grid map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

 

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
"""


class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        counter = 0

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':
                    self.islandVisit(grid, row, col)
                    counter += 1
        return counter

    # time O(n * m)
    # space O(n * m)

    def islandVisit(self, grid, row, col):
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != '1':
            return

        grid[row][col] = '-1'

        self.islandVisit(grid, row, col + 1)
        self.islandVisit(grid, row, col - 1)
        self.islandVisit(grid, row + 1, col)
        self.islandVisit(grid, row - 1, col)


# -----------------------------------------------------------------------
"""
210. Course Schedule II

There are a total of n courses you have to take labelled from 0 to n - 1.

Some courses may have prerequisites, for example, if prerequisites[i] = [ai, bi] this means you must take the course bi before the course ai.

Given the total number of courses numCourses and a list of the prerequisite pairs, return the ordering of courses you should take to finish all courses.

If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].
Example 2:

Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].
Example 3:

Input: numCourses = 1, prerequisites = []
Output: [0]
"""

from collections import defaultdict


class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        n = numCourses
        graph = generate_graph(prerequisites)
        status = defaultdict()
        for i in range(n):
            status[i] = 'not visited'

        res = []
        for i in range(n):
            if status[i] == 'not visited':
                curr = []
                if not get_schedule(graph, status, i, curr):
                    return []
                res += curr

        return res

    # time O(n + e)
    # space O(n + e)


def get_schedule(graph, status, i, curr_res):
    status[i] = 'visiting'
    for adj in graph[i]:
        if status[adj] == 'not visited':
            if not get_schedule(graph, status, adj, curr_res):
                return False
        elif status[adj] == 'visiting':
            return False

    status[i] = 'visited'
    curr_res.append(i)

    return True


def generate_graph(dependencies):
    graph = defaultdict(set)

    for pair in dependencies:
        graph[pair[0]].add(pair[1])

    return graph


# -----------------------------------------------------------------------
"""
222. Count Complete Tree Nodes

Given the root of a complete binary tree, return the number of the nodes in the tree.

According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

 

Example 1:


Input: root = [1,2,3,4,5,6]
Output: 6
Example 2:

Input: root = []
Output: 0
Example 3:

Input: root = [1]
Output: 1
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0

        d = get_depth(root)
        if d == 0:
            return 1

        left, right = 1, pow(2, d) - 1

        while left <= right:
            mid = (left + right) // 2
            if exists_in_tree(mid, d, root):
                left = mid + 1
            else:
                right = mid - 1

        return pow(2, d) - 1 + left

    # time O(d^2)
    # space O(d)


def exists_in_tree(idx, d, root):
    left, right = 0, pow(2, d) - 1

    for _ in range(d):
        mid = (left + right) // 2
        if idx <= mid:
            root = root.left
            right = mid
        else:
            root = root.right
            left = mid + 1

    return root != None


def get_depth(root):
    depth = 0
    while root.left:
        root = root.left
        depth += 1

    return depth


# -----------------------------------------------------------------------
"""
792. Number of Matching Subsequences

Given string S and a dictionary of words words, find the number of words[i] that is a subsequence of S.

Example :
Input: 
S = "abcde"
words = ["a", "bb", "acd", "ace"]
Output: 3
Explanation: There are three words in words that are a subsequence of S: "a", "acd", "ace".
Note:

All words in words and S will only consists of lowercase letters.
The length of S will be in the range of [1, 50000].
The length of words will be in the range of [1, 5000].
The length of words[i] will be in the range of [1, 50].
"""


class Solution(object):
    def numMatchingSubseq(self, S, words):
        """
        :type S: str
        :type words: List[str]
        :rtype: int
        """

        counter = 0
        cached_subsequence = {}

        for word in words:
            if is_subsequence(word, S, cached_subsequence):
                counter += 1

        return counter

    # time O(n * min(m1, m2)) : m1 length of longest word, m2 length of S
    # space O(n * m1)


def is_subsequence(word, string, cached_subsequence):
    if (word, string) in cached_subsequence:
        return cached_subsequence[(word, string)]

    p1, p2 = 0, 0

    while p1 < len(word) and p2 < len(string):
        if word[p1] == string[p2]:
            p1 += 1
            p2 += 1
        else:
            p2 += 1

    res = p1 == len(word)
    cached_subsequence[(word, string)] = res

    return res


class Solution(object):
    def numMatchingSubseq(self, S, words):
        """
        :type S: str
        :type words: List[str]
        :rtype: int
        """

        counter = 0
        head = [[] for _ in range(26)]

        # map each word for its appropriate bucket
        for word in words:
            it = iter(word)
            head[ord(next(it)) - ord('a')].append(it)

        for char in S:
            curr_idx = ord(char) - ord('a')
            curr_bucket = head[curr_idx]
            head[curr_idx] = []

            for it in curr_bucket:
                next_char = next(it, None)

                if next_char:
                    next_idx = ord(next_char) - ord('a')
                    head[next_idx].append(it)
                else:
                    counter += 1

        return counter

        # time O(n * m1 + m2)
        # space O(n * m1)


# -----------------------------------------------------------------------
"""
1423. Maximum Points You Can Obtain from Cards

There are several cards arranged in a row, and each card has an associated number of points The points are given in the integer array cardPoints.

In one step, you can take one card from the beginning or from the end of the row. You have to take exactly k cards.

Your score is the sum of the points of the cards you have taken.

Given the integer array cardPoints and the integer k, return the maximum score you can obtain.

 

Example 1:

Input: cardPoints = [1,2,3,4,5,6,1], k = 3
Output: 12
Explanation: After the first step, your score will always be 1. However, choosing the rightmost card first will maximize your total score. The optimal strategy is to take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.
Example 2:

Input: cardPoints = [2,2,2], k = 2
Output: 4
Explanation: Regardless of which two cards you take, your score will always be 4.
Example 3:

Input: cardPoints = [9,7,7,9,7,7,9], k = 7
Output: 55
Explanation: You have to take all the cards. Your score is the sum of points of all cards.
Example 4:

Input: cardPoints = [1,1000,1], k = 1
Output: 1
Explanation: You cannot take the card in the middle. Your best score is 1. 
Example 5:

Input: cardPoints = [1,79,80,1,1,1,200,1], k = 3
Output: 202
 
"""


class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        points = cardPoints
        n = len(points)

        if k == n:
            return sum(points)

        curr_sum = sum(points[:k])
        max_sum = curr_sum
        left_ptr, right_ptr = k - 1, n - 1

        while left_ptr >= 0:
            curr_sum -= points[left_ptr]
            curr_sum += points[right_ptr]
            max_sum = max(max_sum, curr_sum)

            left_ptr -= 1
            right_ptr -= 1

        return max_sum

    # time O(k)
    # space O(1)


# -----------------------------------------------------------------------
"""
394. Decode String

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

 

Example 1:

Input: s = "3[a]2[bc]"
Output: "aaabcbc"
Example 2:

Input: s = "3[a2[c]]"
Output: "accaccacc"
"""


class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """

        res = []
        for char in s:
            if char == ']':
                add_sequence(res)
            else:
                res.append(char)

        return ''.join(res)

    # time O(n * k * l)
    # space O(n * k * l)


def add_sequence(res):
    curr_seq = get_sequence(res)

    res.pop()  # remove the [

    count = get_count(res)

    curr_seq = curr_seq * count  # decode the sequence

    res += curr_seq


def get_count(res):
    count = []
    while res and res[-1].isnumeric():
        count.append(res.pop())

    count = int(''.join(reversed(count)))

    return count


def get_sequence(res):
    curr_seq = []
    while res and res[-1] != '[':
        curr_seq.append(res.pop())

    if not res or res[-1] != '[':
        raise ValueError('Invalid argument')

    curr_seq.reverse()

    return curr_seq


# -----------------------------------------------------------------------
"""
947. Most Stones Removed with Same Row or Column

On a 2D plane, we place n stones at some integer coordinate points. Each coordinate point may have at most one stone.

A stone can be removed if it shares either the same row or the same column as another stone that has not been removed.

Given an array stones of length n where stones[i] = [xi, yi] represents the location of the ith stone, return the largest possible number of stones that can be removed.

 

Example 1:

Input: stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
Output: 5
Explanation: One way to remove 5 stones is as follows:
1. Remove stone [2,2] because it shares the same row as [2,1].
2. Remove stone [2,1] because it shares the same column as [0,1].
3. Remove stone [1,2] because it shares the same row as [1,0].
4. Remove stone [1,0] because it shares the same column as [0,0].
5. Remove stone [0,1] because it shares the same row as [0,0].
Stone [0,0] cannot be removed since it does not share a row/column with another stone still on the plane.
Example 2:

Input: stones = [[0,0],[0,2],[1,1],[2,0],[2,2]]
Output: 3
Explanation: One way to make 3 moves is as follows:
1. Remove stone [2,2] because it shares the same row as [2,0].
2. Remove stone [2,0] because it shares the same column as [0,0].
3. Remove stone [0,2] because it shares the same row as [0,0].
Stones [0,0] and [1,1] cannot be removed since they do not share a row/column with another stone still on the plane.
Example 3:

Input: stones = [[0,0]]
Output: 0
Explanation: [0,0] is the only stone on the plane, so you cannot remove it.
"""

from collections import defaultdict


class Solution(object):
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """

        # map_ids = defaultdict(int)
        rows, cols = defaultdict(list), defaultdict(list)
        graph = generate_graph(stones, rows, cols)

        visited = set()
        ans = 0
        for ver in graph:
            if ver not in visited:
                ans += count_component(graph, ver, visited) - 1

        return ans

    # time O(n + e) : n num of stones, e num of edges between stones
    # space O(n + e)


def generate_graph(stones, rows, cols):
    graph = defaultdict(set)

    for i, coordinate in enumerate(stones):
        row, col = coordinate

        # add same row adjacents
        for adj in rows[row]:
            graph[i].add(adj)
            graph[adj].add(i)

        # add same col adjacents
        for adj in cols[col]:
            graph[i].add(adj)
            graph[adj].add(i)

        rows[row].append(i)
        cols[col].append(i)

    return graph


def count_component(graph, ver, visited):
    visited.add(ver)

    curr_res = 1

    for adj in graph[ver]:
        if adj not in visited:
            curr_res += count_component(graph, adj, visited)

    return curr_res


"""
1. generate graph between all stones which shared the same row/col
2. count the vertices in the connected components => (count) and add (count -1) to answer

[[0,0], 0
 [0,1], 1
 [1,0], 2
 [1,2], 3
 [2,1], 4
 [2,2]] 5

 rows           
 0: [0, 1]
 1: [2, 3]
 2: [4, 5]

 cols
 0: [0, 2]
 1: [1, 4]
 2: [3, 5]


 *********
 *RR******
 R*R******
 #R*******

 *********
 #*#******
 *#*******
 #*#******

"""

# -----------------------------------------------------------------------
"""
692. Top K Frequent Words

Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.

Example 1:
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
Example 2:
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.
"""

import heapq


class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """

        words_frequency = Counter(words)
        min_heap = [(-freq, word) for word, freq in words_frequency.items()]
        heapq.heapify(min_heap)

        return [heapq.heappop(min_heap)[1] for _ in range(k)]

    # time O(n * log(k))
    # space O(n)


# -----------------------------------------------------------------------
"""
900. RLE Iterator

Write an iterator that iterates through a run-length encoded sequence.

The iterator is initialized by RLEIterator(int[] A), where A is a run-length encoding of some sequence.  More specifically, for all even i, A[i] tells us the number of times that the non-negative integer value A[i+1] is repeated in the sequence.

The iterator supports one function: next(int n), which exhausts the next n elements (n >= 1) and returns the last element exhausted in this way.  If there is no element left to exhaust, next returns -1 instead.

For example, we start with A = [3,8,0,9,2,5], which is a run-length encoding of the sequence [8,8,8,5,5].  This is because the sequence can be read as "three eights, zero nines, two fives".

 

Example 1:

Input: ["RLEIterator","next","next","next","next"], [[[3,8,0,9,2,5]],[2],[1],[1],[2]]
Output: [null,8,8,5,-1]
Explanation: 
RLEIterator is initialized with RLEIterator([3,8,0,9,2,5]).
This maps to the sequence [8,8,8,5,5].
RLEIterator.next is then called 4 times:

.next(2) exhausts 2 terms of the sequence, returning 8.  The remaining sequence is now [8, 5, 5].

.next(1) exhausts 1 term of the sequence, returning 8.  The remaining sequence is now [5, 5].

.next(1) exhausts 1 term of the sequence, returning 5.  The remaining sequence is now [5].

.next(2) exhausts 2 terms, returning -1.  This is because the first term exhausted was 5,
but the second term did not exist.  Since the last term exhausted does not exist, we return -1.
"""


class RLEIterator(object):

    def __init__(self, A):
        """
        :type A: List[int]
        """

        self.nums = A
        self.idx = 0

    def next(self, n):
        """
        :type n: int
        :rtype: int
        """

        while self.idx < len(self.nums) - 1:

            if n <= self.nums[self.idx]:
                self.nums[self.idx] -= n
                return self.nums[self.idx + 1]
            else:
                n -= self.nums[self.idx]
                self.nums[self.idx] = 0
                self.idx += 2

        return -1


# Your RLEIterator object will be instantiated and called as such:
# obj = RLEIterator(A)
# param_1 = obj.next(n)

# -----------------------------------------------------------------------
"""
34. Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

Follow up: Could you write an algorithm with O(log n) runtime complexity?

 

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
"""


class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        left = get_limit(nums, target, 'left')
        right = get_limit(nums, target, 'right')

        return [left, right]

    # time O(log(n))
    # space O(1)


def get_limit(nums, target, limit):
    start, end = 0, len(nums) - 1
    res = -1
    while start <= end:
        mid = (start + end) // 2

        if target < nums[mid]:
            end = mid - 1
        elif target > nums[mid]:
            start = mid + 1
        else:
            res = mid
            if limit == 'left':
                end = mid - 1
            else:
                start = mid + 1

    return res


# -----------------------------------------------------------------------
"""
56. Merge Intervals

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
"""


class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """

        intervals.sort(key=lambda x: x[0])

        res = [intervals[0]]
        n = len(intervals)

        for i in range(1, n):
            curr = intervals[i]
            while res and curr[0] <= res[-1][1]:
                curr = merge_intervals(curr, res[-1])
                res.pop()
            res.append(curr)

        return res

    # time O(n * log(n))
    # space O(n)


def merge_intervals(interval1, interval2):
    left = min(interval1[0], interval2[0])
    right = max(interval1[1], interval2[1])

    return [left, right]


# -----------------------------------------------------------------------
"""
57. Insert Interval

Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

 

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
"""


class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """

        n = len(intervals)
        res = []

        i = 0
        while i < n and intervals[i][0] <= newInterval[0]:
            res.append(intervals[i])
            i += 1

        insert_interval(res, newInterval)

        while i < n:
            insert_interval(res, intervals[i])
            i += 1

        return res

    # time O(n)
    # space O(n)


def insert_interval(res, interval):
    while res and res[-1][1] >= interval[0]:
        interval = merge_intervals(res[-1], interval)
        res.pop()

    res.append(interval)


def merge_intervals(interval1, interval2):
    left = min(interval1[0], interval2[0])
    right = max(interval1[1], interval2[1])

    return [left, right]


# -----------------------------------------------------------------------
"""
528. Random Pick with Weight

You are given an array of positive integers w where w[i] describes the weight of ith index (0-indexed).

We need to call the function pickIndex() which randomly returns an integer in the range [0, w.length - 1]. pickIndex() should return the integer proportional to its weight in the w array. For example, for w = [1, 3], the probability of picking the index 0 is 1 / (1 + 3) = 0.25 (i.e 25%) while the probability of picking the index 1 is 3 / (1 + 3) = 0.75 (i.e 75%).

More formally, the probability of picking index i is w[i] / sum(w).

 

Example 1:

Input
["Solution","pickIndex"]
[[[1]],[]]
Output
[null,0]

Explanation
Solution solution = new Solution([1]);
solution.pickIndex(); // return 0. Since there is only one single element on the array the only option is to return the first element.
Example 2:

Input
["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]]
Output
[null,1,1,1,1,0]

Explanation
Solution solution = new Solution([1, 3]);
solution.pickIndex(); // return 1. It's returning the second element (index = 1) that has probability of 3/4.
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 0. It's returning the first element (index = 0) that has probability of 1/4.

Since this is a randomization problem, multiple answers are allowed so the following outputs can be considered correct :
[null,1,1,1,1,0]
[null,1,1,1,1,1]
[null,1,1,1,0,0]
[null,1,1,1,0,1]
[null,1,0,1,0,0]
......
and so on.
"""
from random import random


class Solution(object):

    def __init__(self, w):
        """
        :type w: List[int]
        """
        self.nums = w
        self.prefix_sum = self.get_prefix_sum()

    def get_prefix_sum(self):
        prefix_sum = []
        curr_sum = 0
        for i, val in enumerate(self.nums):
            curr_sum += val
            prefix_sum.append(curr_sum)

        self.max_num = curr_sum
        return prefix_sum

    # time O(n)
    # space O(n)

    def pickIndex(self):
        """
        :rtype: int
        """

        target = random() * self.max_num

        left, right = 0, len(self.prefix_sum)

        while left < right:
            mid = (left + right) // 2

            if target > self.prefix_sum[mid]:
                left = mid + 1
            else:
                right = mid

        return left

    # time O(log(n))
    # space O(1)


# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()

# -----------------------------------------------------------------------
"""
1631. Path With Minimum Effort

You are a hiker preparing for an upcoming hike. You are given heights, a 2D array of size rows x columns, where heights[row][col] represents the height of cell (row, col). You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right cell, (rows-1, columns-1) (i.e., 0-indexed). You can move up, down, left, or right, and you wish to find a route that requires the minimum effort.

A route's effort is the maximum absolute difference in heights between two consecutive cells of the route.

Return the minimum effort required to travel from the top-left cell to the bottom-right cell.

 

Example 1:



Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
Output: 2
Explanation: The route of [1,3,5,3,5] has a maximum absolute difference of 2 in consecutive cells.
This is better than the route of [1,2,2,2,5], where the maximum absolute difference is 3.
Example 2:



Input: heights = [[1,2,3],[3,8,4],[5,3,5]]
Output: 1
Explanation: The route of [1,2,3,4,5] has a maximum absolute difference of 1 in consecutive cells, which is better than route [1,3,5,3,5].
Example 3:


Input: heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
Output: 0
Explanation: This route does not require any effort.
"""

import heapq


class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """

        rows = len(heights)
        cols = len(heights[0])

        diff_matrix = [[float('inf') for _ in range(cols)] for _ in range(rows)]
        diff_matrix[0][0] = 0

        visited = set()

        min_heap = [(0, 0, 0)]

        while len(min_heap) > 0:

            curr_diff, row, col = heapq.heappop(min_heap)

            visited.add((row, col))

            for n_row, n_col in get_neighbors(row, col):
                if (n_row, n_col) not in visited and is_valid(n_row, n_col, heights):
                    diff = abs(heights[row][col] - heights[n_row][n_col])

                    max_diff = max(diff, curr_diff)

                    if diff_matrix[n_row][n_col] > max_diff:
                        diff_matrix[n_row][n_col] = max_diff
                        heapq.heappush(min_heap, (max_diff, n_row, n_col))

        return diff_matrix[rows - 1][cols - 1]

    # time O(n * m * log(m * n))
    # space O(n * m)


def get_neighbors(row, col):
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    res = []
    for dx, dy in directions:
        res.append((row + dx, col + dy))

    return res


def is_valid(row, col, mat):
    if row < 0 or col < 0 or row >= len(mat) or col >= len(mat[0]):
        return False

    return True


# -----------------------------------------------------------------------
"""
1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

Given an array of integers nums and an integer limit, return the size of the longest non-empty subarray such that 
the absolute difference between any two elements of this subarray is less than or equal to limit.

 

Example 1:

Input: nums = [8,2,4,7], limit = 4
Output: 2 
Explanation: All subarrays are: 
[8] with maximum absolute diff |8-8| = 0 <= 4.
[8,2] with maximum absolute diff |8-2| = 6 > 4. 
[8,2,4] with maximum absolute diff |8-2| = 6 > 4.
[8,2,4,7] with maximum absolute diff |8-2| = 6 > 4.
[2] with maximum absolute diff |2-2| = 0 <= 4.
[2,4] with maximum absolute diff |2-4| = 2 <= 4.
[2,4,7] with maximum absolute diff |2-7| = 5 > 4.
[4] with maximum absolute diff |4-4| = 0 <= 4.
[4,7] with maximum absolute diff |4-7| = 3 <= 4.
[7] with maximum absolute diff |7-7| = 0 <= 4. 
Therefore, the size of the longest subarray is 2.
Example 2:

Input: nums = [10,1,2,4,7,2], limit = 5
Output: 4 
Explanation: The subarray [2,4,7,2] is the longest since the maximum absolute diff is |2-7| = 5 <= 5.
Example 3:

Input: nums = [4,2,2,2,4,4,2,2], limit = 0
Output: 3
"""

from collections import deque


class Solution(object):
    def longestSubarray(self, nums, limit):
        """
        :type nums: List[int]
        :type limit: int
        :rtype: int
        """

        max_q, min_q = deque(), deque()

        left = 0
        res = 0
        for i, num in enumerate(nums):

            add_to_min_q(min_q, num)
            add_to_max_q(max_q, num)

            if max_q[0] - min_q[0] > limit:
                if max_q[0] == nums[left]:
                    max_q.popleft()
                if min_q[0] == nums[left]:
                    min_q.popleft()

                left += 1

            res = max(res, i - left + 1)

        return res

    # time O(n)
    # space O(n)


def add_to_min_q(q, num):
    while q and num < q[-1]:
        q.pop()

    q.append(num)


def add_to_max_q(q, num):
    while q and num > q[-1]:
        q.pop()
    q.append(num)


# -----------------------------------------------------------------------
"""
351. Android Unlock Patterns
Medium

458

801

Add to List

Share
Android devices have a special lock screen with a 3 x 3 grid of dots. Users can set an "unlock pattern" by 
connecting the dots in a specific sequence, forming a series of joined line segments where each segment's endpoints 
are two consecutive dots in the sequence. A sequence of k dots is a valid unlock pattern if both of the following are true:

All the dots in the sequence are distinct.
If the line segment connecting two consecutive dots in the sequence passes through any other dot, the other dot must 
have previously appeared in the sequence. No jumps through non-selected dots are allowed.
Here are some example valid and invalid unlock patterns:



The 1st pattern [4,1,3,6] is invalid because the line connecting dots 1 and 3 pass through dot 2, but dot 2 did not 
previously appear in the sequence.
The 2nd pattern [4,1,9,2] is invalid because the line connecting dots 1 and 9 pass through dot 5, but dot 5 did not 
previously appear in the sequence.
The 3rd pattern [2,4,1,3,6] is valid because it follows the conditions. The line connecting dots 1 and 3 meets the 
condition because dot 2 previously appeared in the sequence.
The 4th pattern [6,5,4,1,9,2] is valid because it follows the conditions. The line connecting dots 1 and 9 meets the 
condition because dot 5 previously appeared in the sequence.
Given two integers m and n, return the number of unique and valid unlock patterns of the Android grid lock screen that 
consist of at least m keys and at most n keys.

Two unlock patterns are considered unique if there is a dot in one sequence that is not in the other, or the order of 
the dots is different.

 

Example 1:

Input: m = 1, n = 1
Output: 9
Example 2:

Input: m = 1, n = 2
Output: 65
"""

start, end = 1, 9


class Solution(object):

    def __init__(self):
        self.has_obstacle = {(1, 3): 2, (1, 7): 4, (1, 9): 5, (2, 8): 5, (3, 7): 5, (3, 1): 2, (3, 9): 6, (4, 6): 5,
                             (6, 4): 5, (7, 1): 4, (7, 3): 5, (7, 9): 8, (8, 2): 5, (9, 7): 8, (9, 3): 6, (9, 1): 5}

    def numberOfPatterns(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """

        if m == n and n == 1:
            return 9

        self.res = 0
        for num in range(start, end + 1):
            self.visited = set()
            self.count_patterns(num, 1, m, n)

        return self.res

    # time O(n^9)
    # space O(n)

    def count_patterns(self, num, count, m, n):

        if count >= m and count <= n:
            self.res += 1

        if count == n:
            return

        self.visited.add(num)

        for next_ in range(start, end + 1):
            if next_ not in self.visited:
                if (num, next_) in self.has_obstacle:
                    if self.has_obstacle[(num, next_)] not in self.visited:
                        continue
                self.count_patterns(next_, count + 1, m, n)

        self.visited.remove(num)


# -----------------------------------------------------------------------
"""
17. Letter Combinations of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.



 

Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
Example 2:

Input: digits = ""
Output: []
Example 3:

Input: digits = "2"
Output: ["a","b","c"]
"""

digits_map = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], '4': ['g', 'h', 'i'], '5': ['j', 'k', 'l'],
              '6': ['m', 'n', 'o'], '7': ['p', 'q', 'r', 's'], '8': ['t', 'u', 'v'], '9': ['w', 'x', 'y', 'z']}


class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """

        if len(digits) == 0:
            return []

        res = []

        generate_letters(digits, 0, [], res)

        return res

    # time O(3^n 4^m)
    # space O(3^n * 4^m)


def generate_letters(digits, idx, curr, res):
    if idx == len(digits):
        res.append(''.join(curr))
        return

    for char in digits_map[digits[idx]]:
        generate_letters(digits, idx + 1, curr + [char], res)


# -----------------------------------------------------------------------
"""
853. Car Fleet

N cars are going to the same destination along a one lane road.  The destination is target miles away.

Each car i has a constant speed speed[i] (in miles per hour), and initial position position[i] miles towards the target along the road.

A car can never pass another car ahead of it, but it can catch up to it, and drive bumper to bumper at the same speed.

The distance between these two cars is ignored - they are assumed to have the same position.

A car fleet is some non-empty set of cars driving at the same position and same speed.  Note that a single car is also a car fleet.

If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.


How many car fleets will arrive at the destination?

 

Example 1:

Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
Output: 3
Explanation:
The cars starting at 10 and 8 become a fleet, meeting each other at 12.
The car starting at 0 doesn't catch up to any other car, so it is a fleet by itself.
The cars starting at 5 and 3 become a fleet, meeting each other at 6.
Note that no other cars meet these fleets before the destination, so the answer is 3.
"""


class Solution(object):
    def carFleet(self, target, position, speed):
        """
        :type target: int
        :type position: List[int]
        :type speed: List[int]
        :rtype: int
        """

        cars = sorted(zip(position, speed))

        times = [float(target - p) / s for p, s in cars]

        res = 0

        while len(times) > 1:
            last = times.pop()

            if last < times[-1]:
                res += 1
            else:
                times[-1] = last

        return res + len(times)

    # time O(n * log(n))
    # space O(n)


# -----------------------------------------------------------------------
"""
304. Range Sum Query 2D - Immutable

Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).

Range Sum Query 2D
The above rectangle (with the red border) is defined by (row1, col1) = (2, 1) and (row2, col2) = (4, 3), which contains sum = 8.

Example:
Given matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
sumRegion(1, 1, 2, 2) -> 11
sumRegion(1, 2, 2, 4) -> 12
"""


class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """

        self.mat = matrix
        self.calc_regions()

    def calc_regions(self):
        n = len(self.mat)
        if n == 0:
            self.regions = [[0, 0]]
            return
        m = len(self.mat[0])

        self.regions = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

        for row in range(1, n + 1):
            for col in range(1, m + 1):
                curr_area = self.regions[row - 1][col] + self.regions[row][col - 1]
                curr_area -= self.regions[row - 1][col - 1]
                curr_area += self.mat[row - 1][col - 1]

                self.regions[row][col] = curr_area

    # time O(n * m)
    # space O(n * m)

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """

        if not self.is_valid(row1, col1) or not self.is_valid(row2, col2):
            return -1

        area = self.regions[row2 + 1][col2 + 1]
        area -= self.regions[row1][col2 + 1]
        area -= self.regions[row2 + 1][col1]
        area += self.regions[row1][col1]

        return area

    # time O(1)
    # space O(1)

    def is_valid(self, row, col):
        if row < 0 or col < 0 or row >= len(self.mat) or col >= len(self.mat[0]):
            return False

        return True


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)


# -----------------------------------------------------------------------
"""
247. Strobogrammatic Number II

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

Find all strobogrammatic numbers that are of length = n.

Example:

Input:  n = 2
Output: ["11","69","88","96"]
"""

opposite = {
    '1': '1',
    '6': '9',
    '9': '6',
    '8': '8',
    '0': '0'
}


class Solution(object):
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

        res = []
        generate_stro_numbers([None] * n, 0, n - 1, res)

        return res

    # time O(5^n)
    # space O(n)


def generate_stro_numbers(curr, start, end, res):
    if start > end:
        res.append(''.join(curr))
        return

    for num in opposite:
        if start == end and num in ('6', '9'):
            continue

        if start != end and start == 0 and num == '0':
            continue

        curr[start], curr[end] = num, opposite[num]

        generate_stro_numbers(curr, start + 1, end - 1, res)


# -----------------------------------------------------------------------
"""
155. Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
 

Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
"""


class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """

        self.nums = []
        self.mins = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """

        self.nums.append(x)
        if not self.mins or x <= self.mins[-1]:
            self.mins.append(x)

    def pop(self):
        """
        :rtype: None
        """
        if not self.nums:
            return

        curr = self.nums.pop()
        if curr == self.mins[-1]:
            self.mins.pop()

    def top(self):
        """
        :rtype: int
        """

        if not self.nums:
            return None

        return self.nums[-1]

    def getMin(self):
        """
        :rtype: int
        """

        if not self.mins:
            return None

        return self.mins[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

# -----------------------------------------------------------------------
"""
380. Insert Delete GetRandom O(1)

Implement the RandomizedSet class:

bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.
Follow up: Could you implement the functions of the class with each function works in average O(1) time?
"""

from random import randint


class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """

        self.nums = []
        self.val_to_idx = {}

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """

        if val in self.val_to_idx:
            return False

        self.nums.append(val)
        self.val_to_idx[val] = len(self.nums) - 1

        return True

    # time O(1)
    # space O(1)

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.val_to_idx or not self.nums:
            return False

        idx = self.val_to_idx[val]

        self.nums[idx], self.nums[-1] = self.nums[-1], self.nums[idx]
        self.nums.pop()
        del self.val_to_idx[val]

        if idx != len(self.nums):
            self.val_to_idx[self.nums[idx]] = idx

        return True

    # time O(1)
    # space O(1)

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """

        if not self.nums:
            return -1

        idx = randint(0, len(self.nums) - 1)

        return self.nums[idx]

    # time O(1)
    # space O(1)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# -----------------------------------------------------------------------
"""
1146. Snapshot Array

Implement a SnapshotArray that supports the following interface:

SnapshotArray(int length) initializes an array-like data structure with the given length.  Initially, each element equals 0.
void set(index, val) sets the element at the given index to be equal to val.
int snap() takes a snapshot of the array and returns the snap_id: the total number of times we called snap() minus 1.
int get(index, snap_id) returns the value at the given index, at the time we took the snapshot with the given snap_id
 

Example 1:

Input: ["SnapshotArray","set","snap","set","get"]
[[3],[0,5],[],[0,6],[0,0]]
Output: [null,null,0,null,5]
Explanation: 
SnapshotArray snapshotArr = new SnapshotArray(3); // set the length to be 3
snapshotArr.set(0,5);  // Set array[0] = 5
snapshotArr.snap();  // Take a snapshot, return snap_id = 0
snapshotArr.set(0,6);
snapshotArr.get(0,0);  // Get the value of array[0] with snap_id = 0, return 5
"""


class SnapshotArray(object):

    def __init__(self, length):
        """
        :type length: int
        """

        self.snap_id = 0
        self.nums = [{0: 0} for _ in range(length)]

    def set(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """

        curr_map = self.nums[index]
        curr_map[self.snap_id] = val

    # time O(1)
    # space O(1)

    def snap(self):
        """
        :rtype: int
        """

        self.snap_id += 1

        return self.snap_id - 1

    # time O(1)
    # space O(1)

    def get(self, index, snap_id):
        """
        :type index: int
        :type snap_id: int
        :rtype: int
        """

        curr_map = self.nums[index]

        # can improve using binary search to find the first valid snap id
        while snap_id not in curr_map:
            snap_id -= 1

        return curr_map.get(snap_id, 0)

    # time O(s_id)
    # space O(1)


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)

# -----------------------------------------------------------------------
"""
1477. Find Two Non-overlapping Sub-arrays Each With Target Sum

Given an array of integers arr and an integer target.

You have to find two non-overlapping sub-arrays of arr each with sum equal target. There can be multiple answers so you have to find an answer where the sum of the lengths of the two sub-arrays is minimum.

Return the minimum sum of the lengths of the two required sub-arrays, or return -1 if you cannot find such two sub-arrays.

 

Example 1:

Input: arr = [3,2,2,4,3], target = 3
Output: 2
Explanation: Only two sub-arrays have sum = 3 ([3] and [3]). The sum of their lengths is 2.
Example 2:

Input: arr = [7,3,4,7], target = 7
Output: 2
Explanation: Although we have three non-overlapping sub-arrays of sum = 7 ([7], [3,4] and [7]), but we will choose the first and third sub-arrays as the sum of their lengths is 2.
"""


class Solution(object):
    def minSumOfLengths(self, arr, target):
        """
        :type arr: List[int]
        :type target: int
        :rtype: int
        """
        n = len(arr)

        min_length = [float('inf') for _ in range(n)]

        curr_sum = 0
        left = 0
        curr_shortest = float('inf')
        res = float('inf')

        for right, val in enumerate(arr):

            curr_sum += val

            while curr_sum > target:
                curr_sum -= arr[left]
                left += 1

            if curr_sum == target:
                if left > 0 and min_length[left - 1] != float('inf'):
                    res = min(res, min_length[left - 1] + right - left + 1)

                curr_shortest = min(curr_shortest, right - left + 1)

            min_length[right] = curr_shortest

        return res if res != float('inf') else -1

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
535. Encode and Decode TinyURL

Note: This is a companion problem to the System Design problem: Design TinyURL.
TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/design-tinyurl and it returns a short URL such as http://tinyurl.com/4e9iAk.

Design the encode and decode methods for the TinyURL service. There is no restriction on how your encode/decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.
"""


class Codec:
    def __init__(self):
        self.id = 0
        self.url_map = {}
        self.base_url = 'http://tinyurl.com/'

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.

        :type longUrl: str
        :rtype: str
        """

        self.url_map[self.id] = longUrl
        url = self.base_url + str(self.id)
        self.id += 1

        return url

    # time O(1)
    # space O(1)

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.

        :type shortUrl: str
        :rtype: str
        """

        id_ = int(shortUrl.replace(self.base_url, ''))
        return self.url_map.get(id_, 0)

    # time O(1)
    # space O(1)


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))

# -----------------------------------------------------------------------
"""
130. Surrounded Regions

Given an m x n matrix board containing 'X' and 'O', capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

 

Example 1:


Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
Explanation: Surrounded regions should not be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
Example 2:

Input: board = [["X"]]
Output: [["X"]]
"""


class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """

        n = len(board)
        m = len(board[0])

        visited = set()

        for row in range(1, n):
            for col in range(1, m):
                if board[row][col] == 'O' and (row, col) not in visited:
                    if is_surrounded(board, row, col, visited):
                        flip(board, row, col)

    # time O(n * m)
    # space O(k) k: num of O's


def is_surrounded(board, row, col, visited):
    if not is_valid(board, row, col):
        return False

    if (row, col) in visited or board[row][col] == 'X':
        return True

    visited.add((row, col))

    res = True
    for n_row, n_col in get_neighbors(row, col):
        if not is_surrounded(board, n_row, n_col, visited):
            res = False

    return res


def is_valid(board, row, col):
    if row < 0 or col < 0 or row >= len(board) or col >= len(board[0]):
        return False

    return True


def get_neighbors(row, col):
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    res = []

    for dx, dy in directions:
        res.append((row + dx, col + dy))

    return res


def flip(board, row, col):
    if board[row][col] == 'X':
        return

    board[row][col] = 'X'

    for n_row, n_col in get_neighbors(row, col):
        flip(board, n_row, n_col)


# -----------------------------------------------------------------------
"""
399. Evaluate Division

You are given an array of variable pairs equations and an array of real numbers values, where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. Each Ai or Bi is a string that represents a single variable.

You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query where you must find the answer for Cj / Dj = ?.

Return the answers to all queries. If a single answer cannot be determined, return -1.0.

Note: The input is always valid. You may assume that evaluating the queries will not result in division by zero and that there is no contradiction.

 

Example 1:

Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
Explanation: 
Given: a / b = 2.0, b / c = 3.0
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
return: [6.0, 0.5, -1.0, 1.0, -1.0 ]
Example 2:

Input: equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
Output: [3.75000,0.40000,5.00000,0.20000]
"""

from collections import defaultdict, deque


class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight


class Graph:
    def __init__(self):
        self.edges = defaultdict(set)
        self.vertices = set()

    def add_edge(self, u, v, weight):
        edge = Edge(u, v, weight)
        rev_edge = Edge(v, u, 1 / weight)
        self.edges[u].add(edge)
        self.edges[v].add(rev_edge)
        self.vertices.add(u)
        self.vertices.add(v)

    def calc_equation(self, u, v):
        if u not in self.vertices or v not in self.vertices:
            return -1

        if u == v:
            return 1

        #         visited = set()
        #         res = self.calc_res(u, v, visited)

        res = self.calc_res(u, v)

        if res == -1:
            return res

        self.add_edge(u, v, res)

        return res

    # BFS
    def calc_res(self, u, v):

        q = deque([(u, 1)])
        visited = set([u])

        while q:
            curr, curr_res = q.popleft()

            for neigh in self.edges[curr]:
                if neigh.v in visited:
                    continue

                if neigh.v == v:
                    return curr_res * neigh.weight

                visited.add(neigh.v)
                q.append((neigh.v, neigh.weight * curr_res))

        return -1

    # DFS


#     def calc_res(self, u, v, visited):

#         visited.add(u)
#         res = 1
#         for neigh in self.edges[u]:
#             if neigh.v in visited:
#                 continue

#             res *= neigh.weight
#             if v == neigh.v:
#                 return res

#             calc = self.calc_res(neigh.v, v, visited)
#             if calc != -1:
#                 return res * calc
#             res /= neigh.weight

#         return -1


class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]

        """

        graph = Graph()
        for i, pair in enumerate(equations):
            u, v = pair
            graph.add_edge(u, v, values[i])

        res = []
        for u, v in queries:
            res.append(graph.calc_equation(u, v))

        return res

    # time O(n + q * n)
    # space O(n + q)


# -----------------------------------------------------------------------
"""
5. Longest Palindromic Substring

Given a string s, return the longest palindromic substring in s.

 
Example 1:

Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
Example 3:

Input: s = "a"
Output: "a"
Example 4:

Input: s = "ac"
Output: "a"
"""


class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """

        if not s:
            return ''

        start, end = 0, 0
        max_len = 0
        for i in range(len(s)):

            # odd pal
            len1 = expand_around_center(s, i, i)
            # even pal
            len2 = expand_around_center(s, i, i + 1)
            new_len = max(len1, len2)

            if new_len > max_len:
                max_len = new_len
                mid = max_len // 2
                if max_len % 2 == 0:
                    start = i - mid + 1
                else:
                    start = i - mid

                end = i + mid

        return s[start: end + 1]

    # time O(n^2)
    # space O(1)


def expand_around_center(s, start, end):
    while start >= 0 and end < len(s) and s[start] == s[end]:
        start -= 1
        end += 1

    return end - start - 1


def lps(s, idx1, idx2, memo):
    if idx1 == idx2:
        memo[idx1][idx2] = 1
        return 1

    if idx2 < idx1:
        return 0

    if memo[idx1][idx2] is not None:
        return memo[idx1][idx2]

    res = 0
    if s[idx1] == s[idx2]:
        ret = lps(s, idx1 + 1, idx2 - 1, memo)
        if ret == idx2 - idx1 - 1:
            res = ret + 2
    else:
        res = max(lps(s, idx1 + 1, idx2, memo), lps(s, idx1, idx2 - 1, memo), lps(s, idx1 + 1, idx2 - 1, memo))

    memo[idx1][idx2] = res

    return res


"""
lps[i...j] = { if s[i] == s[j]: if lps[i+1... j-1] == j-i: i - j - 1,
                o.w: max(lps[i...j-1], lps[i+1...j], lps[i+1...j-1])
             }
"""
# -----------------------------------------------------------------------
"""
322. Coin Change

You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Example 3:

Input: coins = [1], amount = 0
Output: 0
"""


class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        memo = [0 for _ in range(amount + 1)]

        return count_coins(coins, amount, memo)

    # time O(n * a)
    # space O(a)

    # bottom up


#         coins_amount = [float('inf') for _ in range(amount + 1)]
#         coins_amount[0] = 0

#         for coin in coins:
#             for curr_amount in range(coin, amount + 1):
#                 coins_amount[curr_amount] = min(coins_amount[curr_amount], coins_amount[curr_amount - coin] + 1)

#         return coins_amount[amount] if coins_amount[amount] != float('inf') else -1


def count_coins(coins, amount, memo):
    if amount < 0:
        return -1

    if amount == 0:
        return 0

    if memo[amount] != 0:
        return memo[amount]

    curr_min = float('inf')

    for coin in coins:
        curr = count_coins(coins, amount - coin, memo)
        if curr >= 0 and curr < curr_min:
            curr_min = curr + 1

    memo[amount] = -1 if curr_min == float('inf') else curr_min

    return memo[amount]


# -----------------------------------------------------------------------
"""
299. Bulls and Cows

You are playing the Bulls and Cows game with your friend.

You write down a secret number and ask your friend to guess what the number is. When your friend makes a guess, you provide a hint with the following info:

The number of "bulls", which are digits in the guess that are in the correct position.
The number of "cows", which are digits in the guess that are in your secret number but are located in the wrong position. Specifically, the non-bull digits in the guess that could be rearranged such that they become bulls.
Given the secret number secret and your friend's guess guess, return the hint for your friend's guess.

The hint should be formatted as "xAyB", where x is the number of bulls and y is the number of cows. Note that both secret and guess may contain duplicate digits.

 

Example 1:

Input: secret = "1807", guess = "7810"
Output: "1A3B"
Explanation: Bulls are connected with a '|' and cows are underlined:
"1807"
  |
"7810"
Example 2:

Input: secret = "1123", guess = "0111"
Output: "1A1B"
Explanation: Bulls are connected with a '|' and cows are underlined:
"1123"        "1123"
  |      or     |
"0111"        "0111"
Note that only one of the two unmatched 1s is counted as a cow since the non-bull digits can only be rearranged to allow one 1 to be a bull.
"""

from collections import defaultdict


class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """

        x = get_num_of_hits(secret, guess)
        y = get_num_of_miss(secret, guess)

        return '{}A{}B'.format(x, y)

    # time O(A + B)
    # space O(A + B)


def get_num_of_hits(number, guess):
    count = 0

    for i in range(len(number)):
        if number[i] == guess[i]:
            count += 1

    return count


def get_num_of_miss(number, guess):
    num_digs = digits_to_idx(number)
    guess_digs = digits_to_idx(guess)

    count = 0
    for i in range(len(number)):
        if number[i] != guess[i]:
            for idx in guess_digs[number[i]]:
                if guess[idx] != number[idx]:
                    count += 1
                    guess_digs[number[i]].remove(idx)
                    break

    return count


def digits_to_idx(num):
    digits = defaultdict(list)

    for i, dig in enumerate(num):
        digits[dig].append(i)

    return digits


# -----------------------------------------------------------------------
"""
152. Maximum Product Subarray

Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

It is guaranteed that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
"""


class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if len(nums) < 1:
            return 0

        min_val = max_val = res = nums[0]

        for i in range(1, len(nums)):
            val = nums[i]
            temp = min(val, min_val * val, max_val * val)
            max_val = max(val, min_val * val, max_val * val)

            min_val = temp

            res = max(res, max_val)

        return res

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
939. Minimum Area Rectangle

Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points, with sides parallel to the x and y axes.

If there isn't any rectangle, return 0.

 

Example 1:

Input: [[1,1],[1,3],[3,1],[3,3],[2,2]]
Output: 4
Example 2:

Input: [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]
Output: 2
"""


class Solution(object):
    def minAreaRect(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """

        seen = set()
        min_area = float('inf')

        for x1, y1 in points:

            for x2, y2 in seen:
                if (x1, y2) in seen and (x2, y1) in seen:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    min_area = min(min_area, area)

            seen.add((x1, y1))

        return min_area if min_area != float('inf') else 0

    # time O(n^2)
    # space O(n)


# -----------------------------------------------------------------------
"""
426. Convert Binary Search Tree to Sorted Doubly Linked List

Convert a Binary Search Tree to a sorted Circular Doubly-Linked List in place.

You can think of the left and right pointers as synonymous to the predecessor and successor pointers in a doubly-linked list. For a circular doubly linked list, the predecessor of the first element is the last element, and the successor of the last element is the first element.

We want to do the transformation in place. After the transformation, the left pointer of the tree node should point to its predecessor, and the right pointer should point to its successor. You should return the pointer to the smallest element of the linked list.
"""

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""


class Solution(object):
    def treeToDoublyList(self, root):
        """
        :type root: Node
        :rtype: Node
        """

        if not root:
            return root

        left = self.treeToDoublyList(root.left)
        right = self.treeToDoublyList(root.right)

        root.left = root.right = root

        left = concatenate(left, root)

        return concatenate(left, right)

    # time O(n)
    # space O(1)


def concatenate(left, right):
    if not left:
        return right

    if not right:
        return left

    left_last = left.left
    right_last = right.left

    left_last.right = right
    right.left = left_last

    left.left = right_last
    right_last.right = left

    return left


# -----------------------------------------------------------------------
"""
92. Reverse Linked List II

Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

 

Example 1:


Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
Example 2:

Input: head = [5], left = 1, right = 1
Output: [5]
"""


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """

        if not head or not head.next:
            return head

        count = right - left
        left_p, right_p = head, head

        while right_p and count > 0:
            right_p = right_p.next
            count -= 1

        count = 1

        prev = left_p
        while count < left:
            prev = left_p
            left_p = left_p.next
            right_p = right_p.next
            count += 1

        if left_p == head:
            prev = None
        else:
            prev.next = None

        next_p = right_p.next
        right_p.next = None

        r_head, tail = reverse(left_p)

        if prev:
            prev.next = r_head
        tail.next = next_p

        return head if prev else r_head

    # time O(n)
    # space O(1)


def reverse(head):
    curr = head
    prev = None

    while curr:
        next_ = curr.next
        curr.next = prev
        prev = curr
        curr = next_

    return prev, head


# -----------------------------------------------------------------------
"""
1057. Campus Bikes

On a campus represented as a 2D grid, there are N workers and M bikes, with N <= M. Each worker and bike is a 2D coordinate on this grid.

Our goal is to assign a bike to each worker. Among the available bikes and workers, we choose the (worker, bike) pair with the shortest Manhattan distance between each other, and assign the bike to that worker. (If there are multiple (worker, bike) pairs with the same shortest Manhattan distance, we choose the pair with the smallest worker index; if there are multiple ways to do that, we choose the pair with the smallest bike index). We repeat this process until there are no available workers.

The Manhattan distance between two points p1 and p2 is Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|.

Return a vector ans of length N, where ans[i] is the index (0-indexed) of the bike that the i-th worker is assigned to.

 

Example 1:



Input: workers = [[0,0],[2,1]], bikes = [[1,2],[3,3]]
Output: [1,0]
Explanation: 
Worker 1 grabs Bike 0 as they are closest (without ties), and Worker 0 is assigned Bike 1. So the output is [1, 0].
Example 2:



Input: workers = [[0,0],[1,1],[2,0]], bikes = [[1,0],[2,2],[2,1]]
Output: [0,2,1]
Explanation: 
Worker 0 grabs Bike 0 at first. Worker 1 and Worker 2 share the same distance to Bike 2, thus Worker 1 is assigned to Bike 2, and Worker 2 will take Bike 1. So the output is [0,2,1].
 

Note:

0 <= workers[i][j], bikes[i][j] < 1000
All worker and bike locations are distinct.
1 <= workers.length <= bikes.length <= 1000
"""


class Solution(object):
    def assignBikes(self, workers, bikes):
        """
        :type workers: List[List[int]]
        :type bikes: List[List[int]]
        :rtype: List[int]
        """

        distances = get_distances(workers, bikes)

        visited = set()
        res = [-1] * len(workers)

        for d, i, j in distances:

            if res[i] == -1 and j not in visited:
                res[i] = j
                visited.add(j)

        return res

    # time O(n*m * log(n*m))
    # space O(n*m)


def get_distances(workers, bikes):
    distances = []

    for i in range(len(workers)):
        for j in range(len(bikes)):
            distances.append((distance(workers[i], bikes[j]), i, j))

    return sorted(distances)


def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


"""

(weight, from, to)

"""

import heapq


class Solution(object):
    def assignBikes(self, workers, bikes):
        """
        :type workers: List[List[int]]
        :type bikes: List[List[int]]
        :rtype: List[int]
        """

        n = len(workers)
        distances = get_distances(workers, bikes)

        res = [-1] * n

        min_heap = []
        for i in range(n):
            min_heap.append(distances[i].pop())

        heapq.heapify(min_heap)
        used_bikes = set()

        while min_heap and len(used_bikes) < n:
            distance, worker, bike = heapq.heappop(min_heap)

            if res[worker] != -1:
                continue

            if bike not in used_bikes:
                res[worker] = bike
                used_bikes.add(bike)
            else:
                heapq.heappush(min_heap, distances[worker].pop())

        return res

    # time O(n * m)
    # space O(n * m)


def get_distances(workers, bikes):
    distances = []

    for i in range(len(workers)):
        curr = []
        for j in range(len(bikes)):
            curr.append((distance(workers[i], bikes[j]), i, j))

        distances.append(sorted(curr, reverse=True))

    return distances


def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# -----------------------------------------------------------------------
"""
562. Longest Line of Consecutive One in Matrix

Given a 01 matrix M, find the longest line of consecutive one in the matrix. The line could be horizontal, vertical, diagonal or anti-diagonal.
Example:
Input:
[[0,1,1,0],
 [0,1,1,0],
 [0,0,0,1]]
Output: 3
Hint: The number of elements in the given matrix will not exceed 10,000.
"""


class Solution(object):
    def longestLine(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        n = len(M)
        if n == 0:
            return 0

        m = len(M[0])

        length = [[(0, 0, 0, 0) for _ in range(m + 2)] for _ in range(n + 1)]

        max_length = 0
        for row in range(1, n + 1):
            for col in range(1, m + 1):
                if M[row - 1][col - 1] == 0:
                    continue

                up = length[row - 1][col][1]
                left = length[row][col - 1][0]
                diagonal = length[row - 1][col - 1][2]
                anti_diagonal = length[row - 1][col + 1][3]
                length[row][col] = (left + 1, up + 1, diagonal + 1, anti_diagonal + 1)

                max_length = max(max_length, max(length[row][col]))

        return max_length

    # time O(n * m)
    # space O(n * m)


# -----------------------------------------------------------------------
"""
593. Valid Square

Given the coordinates of four points in 2D space p1, p2, p3 and p4, return true if the four points construct a square.

The coordinate of a point pi is represented as [xi, yi]. The input is not given in any order.

A valid square has four equal sides with positive length and four equal angles (90-degree angles).

 

Example 1:

Input: p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,1]
Output: true
Example 2:

Input: p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,12]
Output: false
Example 3:

Input: p1 = [1,0], p2 = [-1,0], p3 = [0,1], p4 = [0,-1]
Output: true
"""


class Solution(object):
    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """

        points = sorted([p1, p2, p3, p4])

        for i in range(len(points)):
            for k in range(i + 1, len(points)):
                for h in range(k + 1, len(points)):
                    p1 = points[i]
                    p2 = points[k]
                    p3 = points[h]
                    if p1 == p2 or p1 == p3:
                        continue

                    if not is_vertical(p1, p2, p3):
                        continue

                    x = p2[0] + p3[0] - p1[0]
                    y = p2[1] + p3[1] - p1[1]
                    p4 = [x, y]

                    if p4 not in points or p4 == p1 or p4 == p2 or p4 == p3:
                        continue

                    if distance(p1, p2) == distance(p2, p4) and distance(p1, p2) == distance(p1, p3):
                        return True

        return False


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = (x1 - x2) ** 2
    dy = (y1 - y2) ** 2

    return (dx + dy) ** 0.5


def is_vertical(p1, p2, p3):
    dx12 = p2[0] - p1[0]
    dy12 = p2[1] - p1[1]
    dx13 = p3[0] - p1[0]
    dy13 = p3[1] - p1[1]

    return dy12 * dy13 + dx12 * dx13 == 0


# def slope(p1, p2):
#     x1, y1 = p1
#     x2, y2 = p2

#     if x1 == x2:
#         return 0

#     return (y2 - y1) / (x2 - x1)


"""

p1, p2, p3
p1 make 90 degree with p2 and p3

p4 => p3.x + p2.x - p1.x, p3.y + p2.y - p1.y


"""
# -----------------------------------------------------------------------
"""
1066. Campus Bikes II

On a campus represented as a 2D grid, there are N workers and M bikes, with N <= M. Each worker and bike is a 2D coordinate on this grid.

We assign one unique bike to each worker so that the sum of the Manhattan distances between each worker and their assigned bike is minimized.

The Manhattan distance between two points p1 and p2 is Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|.

Return the minimum possible sum of Manhattan distances between each worker and their assigned bike.

 

Example 1:


Input: workers = [[0,0],[2,1]], bikes = [[1,2],[3,3]]
Output: 6
Explanation: 
We assign bike 0 to worker 0, bike 1 to worker 1. The Manhattan distance of both assignments is 3, so the output is 6.
Example 2:


Input: workers = [[0,0],[1,1],[2,0]], bikes = [[1,0],[2,2],[2,1]]
Output: 4
Explanation: 
We first assign bike 0 to worker 0, then assign bike 1 to worker 1 or worker 2, bike 2 to worker 2 or worker 1. Both assignments lead to sum of the Manhattan distances as 4.
Example 3:

Input: workers = [[0,0],[1,0],[2,0],[3,0],[4,0]], bikes = [[0,999],[1,999],[2,999],[3,999],[4,999]]
Output: 4995
"""


class Solution(object):
    def assignBikes(self, workers, bikes):
        """
        :type workers: List[List[int]]
        :type bikes: List[List[int]]
        :rtype: int
        """

        return dfs(workers, bikes, 0, 0, {})

    # time O(n * m)
    # space O(n * m)


def dfs(workers, bikes, idx, mask, memo):
    if idx == len(workers):
        return 0

    if (idx, mask) in memo:
        return memo[(idx, mask)]

    worker = workers[idx]
    res = float('inf')

    for i in range(len(bikes)):

        if (1 << i) & mask == 0:
            distance = get_distance(worker, bikes[i])
            res = min(res, dfs(workers, bikes, idx + 1, (1 << i) | mask, memo) + distance)

    memo[(idx, mask)] = res

    return res


def get_distance(worker, bike):
    w_x, w_y = worker
    b_x, b_y = bike

    return abs(b_x - w_x) + abs(b_y - w_y)


# -----------------------------------------------------------------------
"""
1618. Maximum Font to Fit a Sentence in a Screen

You are given a string text. We want to display text on a screen of width w and height h. You can choose any font size from array fonts, which contains the available font sizes in ascending order.

You can use the FontInfo interface to get the width and height of any character at any available font size.

The FontInfo interface is defined as such:

interface FontInfo {
  // Returns the width of character ch on the screen using font size fontSize.
  // O(1) per call
  public int getWidth(int fontSize, char ch);

  // Returns the height of any character on the screen using font size fontSize.
  // O(1) per call
  public int getHeight(int fontSize);
}
The calculated width of text for some fontSize is the sum of every getWidth(fontSize, text[i]) call for each 0 <= i < text.length (0-indexed). The calculated height of text for some fontSize is getHeight(fontSize). Note that text is displayed on a single line.

It is guaranteed that FontInfo will return the same value if you call getHeight or getWidth with the same parameters.

It is also guaranteed that for any font size fontSize and any character ch:

getHeight(fontSize) <= getHeight(fontSize+1)
getWidth(fontSize, ch) <= getWidth(fontSize+1, ch)
Return the maximum font size you can use to display text on the screen. If text cannot fit on the display with any font size, return -1.

 

Example 1:

Input: text = "helloworld", w = 80, h = 20, fonts = [6,8,10,12,14,16,18,24,36]
Output: 6
Example 2:

Input: text = "leetcode", w = 1000, h = 50, fonts = [1,2,4]
Output: 4
Example 3:

Input: text = "easyquestion", w = 100, h = 100, fonts = [10,15,20,25]
Output: -1
"""


# """
# This is FontInfo's API interface.
# You should not implement it, or speculate about its implementation
# """
# class FontInfo(object):
#    Return the width of char ch when fontSize is used.
#    def getWidth(self, fontSize, ch):
#        """
#        :type fontSize: int
#        :type ch: char
#        :rtype int
#        """
#
#    def getHeight(self, fontSize):
#        """
#        :type fontSize: int
#        :rtype int
#        """

class Solution(object):
    def maxFont(self, text, w, h, fonts, fontInfo):
        """
        :type text: str
        :type w: int
        :type h: int
        :type fonts: List[int]
        :type fontInfo: FontInfo
        :rtype: int
        """

        left, right = 0, len(fonts) - 1
        res = -1

        while left <= right:
            mid = (left + right) >> 1

            if can_font_fit(text, w, h, fonts[mid], fontInfo):
                res = fonts[mid]
                left = mid + 1
            else:
                right = mid - 1

        return res

    # time O(n log(m))
    # space O(1)


def can_font_fit(text, w, h, curr_font, fontInfo):
    if fontInfo.getHeight(curr_font) > h:
        return False

    if get_width(text, curr_font, fontInfo) > w:
        return False

    return True


def get_width(text, curr_font, fontInfo):
    width = 0
    for c in text:
        width += fontInfo.getWidth(curr_font, c)

    return width


# -----------------------------------------------------------------------
"""
801. Minimum Swaps To Make Sequences Increasing

We have two integer sequences A and B of the same non-zero length.

We are allowed to swap elements A[i] and B[i].  Note that both elements are in the same index position in their respective sequences.

At the end of some number of swaps, A and B are both strictly increasing.  (A sequence is strictly increasing if and only if A[0] < A[1] < A[2] < ... < A[A.length - 1].)

Given A and B, return the minimum number of swaps to make both sequences strictly increasing.  It is guaranteed that the given input always makes it possible.

Example:
Input: A = [1,3,5,4], B = [1,2,3,7]
Output: 1
Explanation: 
Swap A[3] and B[3].  Then the sequences are:
A = [1, 3, 5, 7] and B = [1, 2, 3, 4]
which are both strictly increasing.
Note:

A, B are arrays with the same length, and that length will be in the range [1, 1000].
A[i], B[i] are integer values in the range [0, 2000].
"""


class Solution(object):
    def minSwap(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """

        n = len(A)
        with_swap = [1] * n
        no_swap = [0] * n

        for i in range(1, n):
            with_swap[i] = n
            no_swap[i] = n

            if A[i - 1] < A[i] and B[i - 1] < B[i]:
                no_swap[i] = no_swap[i - 1]
                with_swap[i] = with_swap[i - 1] + 1

            if A[i - 1] < B[i] and B[i - 1] < A[i]:
                no_swap[i] = min(no_swap[i], with_swap[i - 1])
                with_swap[i] = min(with_swap[i], no_swap[i - 1] + 1)

        return min(with_swap[-1], no_swap[-1])

        # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
1027. Longest Arithmetic Subsequence

Given an array A of integers, return the length of the longest arithmetic subsequence in A.

Recall that a subsequence of A is a list A[i_1], A[i_2], ..., A[i_k] with 0 <= i_1 < i_2 < ... < i_k <= A.length - 1, and that a sequence B is arithmetic if B[i+1] - B[i] are all the same value (for 0 <= i < B.length - 1).

 

Example 1:

Input: A = [3,6,9,12]
Output: 4
Explanation: 
The whole array is an arithmetic sequence with steps of length = 3.
Example 2:

Input: A = [9,4,7,2,10]
Output: 3
Explanation: 
The longest arithmetic subsequence is [4,7,10].
Example 3:

Input: A = [20,1,15,3,10,5,8]
Output: 4
Explanation: 
The longest arithmetic subsequence is [20,15,10,5].
"""

from collections import defaultdict


class Solution(object):
    def longestArithSeqLength(self, A):
        """
        :type A: List[int]
        :rtype: int
        """

        arith_length = defaultdict()
        n = len(A)
        res = 0
        for i in range(n):
            curr = defaultdict()
            for j in range(i):
                diff = A[i] - A[j]

                if diff in arith_length[j]:
                    curr[diff] = arith_length[j][diff] + 1
                else:
                    curr[diff] = 2

            arith_length[i] = curr
            if curr:
                res = max(res, max(curr.values()))

        return res

    # time O(n^2)
    # space O(n)


# -----------------------------------------------------------------------
"""
1197. Minimum Knight Moves

In an infinite chess board with coordinates from -infinity to +infinity, you have a knight at square [0, 0].

A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.



Return the minimum number of steps needed to move the knight to the square [x, y].  It is guaranteed the answer exists.

 

Example 1:

Input: x = 2, y = 1
Output: 1
Explanation: [0, 0] → [2, 1]
Example 2:

Input: x = 5, y = 5
Output: 4
Explanation: [0, 0] → [2, 1] → [4, 2] → [3, 4] → [5, 5]
"""

from collections import deque


class Solution(object):
    def minKnightMoves(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """

        x = abs(x)
        y = abs(y)

        queue = deque([(0, 0, 0)])
        visited = set()
        visited.add((0, 0))

        while queue:

            distance, curr_x, curr_y = queue.popleft()

            if curr_x == x and curr_y == y:
                return distance

            for n_x, n_y in get_neighbors(curr_x, curr_y):
                if (n_x, n_y) in visited:
                    continue

                if not is_valid(n_x, n_y, x, y):
                    continue

                queue.append((distance + 1, n_x, n_y))
                visited.add((n_x, n_y))

        return -1

        # time O(n/4)
        # space O(n/4)


def is_valid(n_x, n_y, x, y):
    return -2 <= n_x <= x + 2 and -2 <= n_y <= y + 2


def get_neighbors(x, y):
    dirs = [[-1, 2], [-2, 1], [-1, -2], [-2, -1], [1, 2], [2, 1], [2, -1], [1, -2]]
    res = []

    for dx, dy in dirs:
        res.append((x + dx, y + dy))

    return res


# -----------------------------------------------------------------------
"""
855. Exam Room

In an exam room, there are N seats in a single row, numbered 0, 1, 2, ..., N-1.

When a student enters the room, they must sit in the seat that maximizes the distance to the closest person.  If there are multiple such seats, they sit in the seat with the lowest number.  (Also, if no one is in the room, then the student sits at seat number 0.)

Return a class ExamRoom(int N) that exposes two functions: ExamRoom.seat() returning an int representing what seat the student sat in, and ExamRoom.leave(int p) representing that the student in seat number p now leaves the room.  It is guaranteed that any calls to ExamRoom.leave(p) have a student sitting in seat p.

 

Example 1:

Input: ["ExamRoom","seat","seat","seat","seat","leave","seat"], [[10],[],[],[],[],[4],[]]
Output: [null,0,9,4,2,null,5]
Explanation:
ExamRoom(10) -> null
seat() -> 0, no one is in the room, then the student sits at seat number 0.
seat() -> 9, the student sits at the last seat number 9.
seat() -> 4, the student sits at the last seat number 4.
seat() -> 2, the student sits at the last seat number 2.
leave(4) -> null
seat() -> 5, the student sits at the last seat number 5.
"""


class ExamRoom(object):

    def __init__(self, N):
        """
        :type N: int
        """
        self.seats = []
        self.N = N

    def seat(self):
        """
        :rtype: int
        """
        if not self.seats:
            self.seats.append(0)
            return 0

        dist, position = self.seats[0], 0

        for i, s in enumerate(self.seats):
            if i:
                prev = self.seats[i - 1]

                curr_dist = (s - prev) // 2
                if curr_dist > dist:
                    dist = curr_dist
                    position = prev + curr_dist

        curr_dist = self.N - self.seats[-1] - 1
        if curr_dist > dist:
            position = self.N - 1

        bisect.insort(self.seats, position)
        return position

        # time O(n)

    # space O(1)

    def leave(self, p):
        """
        :type p: int
        :rtype: None
        """
        if p < 0 or p >= self.N:
            return

        self.seats.remove(p)

    # time O(1)
    # space O(1)


# Your ExamRoom object will be instantiated and called as such:
# obj = ExamRoom(N)
# param_1 = obj.seat()
# obj.leave(p)
# -----------------------------------------------------------------------
"""
1504. Count Submatrices With All Ones

Given a rows * columns matrix mat of ones and zeros, return how many submatrices have all ones.

 

Example 1:

Input: mat = [[1,0,1],
              [1,1,0],
              [1,1,0]]
Output: 13
Explanation:
There are 6 rectangles of side 1x1.
There are 2 rectangles of side 1x2.
There are 3 rectangles of side 2x1.
There is 1 rectangle of side 2x2. 
There is 1 rectangle of side 3x1.
Total number of rectangles = 6 + 2 + 3 + 1 + 1 = 13.
Example 2:

Input: mat = [[0,1,1,0],
              [0,1,1,1],
              [1,1,1,0]]
Output: 24
Explanation:
There are 8 rectangles of side 1x1.
There are 5 rectangles of side 1x2.
There are 2 rectangles of side 1x3. 
There are 4 rectangles of side 2x1.
There are 2 rectangles of side 2x2. 
There are 2 rectangles of side 3x1. 
There is 1 rectangle of side 3x2. 
Total number of rectangles = 8 + 5 + 2 + 4 + 2 + 2 + 1 = 24.
Example 3:

Input: mat = [[1,1,1,1,1,1]]
Output: 21
Example 4:

Input: mat = [[1,0,1],[0,1,0],[1,0,1]]
Output: 5

"""


class Solution(object):
    def numSubmat(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """

        n = len(mat)
        m = len(mat[0])
        res = 0

        heights = [0] * m
        for row in range(n):
            for col in range(m):
                if mat[row][col] == 0:
                    heights[col] = 0
                else:
                    heights[col] += 1
            res += count_sub_mat(heights)

        return res

        # time O(n * m)
        # space O(m)

        # solution 1
        for up in range(n):
            heights = [1] * m
            for down in range(up, n):
                for k in range(m):
                    if mat[down][k] == 0:
                        heights[k] = 0

                res += count_ones(heights)

        return res

    # time O(n^2 * m)
    # space O(m)


def count_ones(arr):
    res, length = 0, 0

    for val in arr:
        if val == 0:
            length = 0
            continue
        length += 1
        res += length

    return res


def count_sub_mat(heights):
    sum_ = [0] * len(heights)
    stack = []

    res = 0
    for i, val in enumerate(heights):

        while stack and heights[stack[-1]] >= val:
            stack.pop()

        if stack:
            prev_idx = stack[-1]
            sum_[i] = sum_[prev_idx] + (val * (i - prev_idx))
        else:
            sum_[i] = val * (i + 1)

        res += sum_[i]
        stack.append(i)

    return res


# -----------------------------------------------------------------------
"""
1673. Find the Most Competitive Subsequence

Given an integer array nums and a positive integer k, return the most competitive subsequence of nums of size k.

An array's subsequence is a resulting sequence obtained by erasing some (possibly zero) elements from the array.

We define that a subsequence a is more competitive than a subsequence b (of the same length) if in the first position where a and b differ, subsequence a has a number less than the corresponding number in b. For example, [1,3,4] is more competitive than [1,3,5] because the first position they differ is at the final number, and 4 is less than 5.

 

Example 1:

Input: nums = [3,5,2,6], k = 2
Output: [2,6]
Explanation: Among the set of every possible subsequence: {[3,5], [3,2], [3,6], [5,2], [5,6], [2,6]}, [2,6] is the most competitive.
Example 2:

Input: nums = [2,4,3,3,5,4,9,6], k = 4
Output: [2,3,3,4]
"""


class Solution(object):
    def mostCompetitive(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """

        stack = []
        for i, val in enumerate(nums):
            while stack and stack[-1] > val and len(stack) - 1 + len(nums) - i >= k:
                stack.pop()
            if len(stack) < k:
                stack.append(val)
        return stack

    # time O(n)
    # space O(k)


# -----------------------------------------------------------------------
"""
361. Bomb Enemy

Given an m x n matrix grid where each cell is either a wall 'W', an enemy 'E' or empty '0', return the maximum enemies you can kill using one bomb. You can only place the bomb in an empty cell.

The bomb kills all the enemies in the same row and column from the planted point until it hits the wall since it is too strong to be destroyed.

 

Example 1:


Input: grid = [["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]
Output: 3
Example 2:


Input: grid = [["W","W","W"],["0","0","0"],["E","E","E"]]
Output: 1
 
"""


class Solution(object):
    def maxKilledEnemies(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        n = len(grid)
        m = len(grid[0])

        enemies_count = count_enemies(grid)

        res = 0

        for row in range(n):
            for col in range(m):
                if grid[row][col] == 'E' or grid[row][col] == 'W':
                    continue

                res = max(res, enemies_count[row][col])

        return res

    # time O(n * m)
    # space O(n * m)


def count_enemies(grid):
    n = len(grid)
    m = len(grid[0])

    enemies_count = [[0 for _ in range(m)] for _ in range(n)]

    for r in range(n):
        curr = 0
        last_wall = 0
        for col in range(m):
            if grid[r][col] == 'E':
                curr += 1
            elif grid[r][col] == '0':
                continue
            else:
                for j in range(last_wall, col):
                    enemies_count[r][j] += curr
                last_wall = col + 1
                curr = 0

        for j in range(last_wall, col + 1):
            enemies_count[r][j] += curr

    for c in range(m):
        curr = 0
        last_wall = 0
        for row in range(n):
            if grid[row][c] == 'E':
                curr += 1
            elif grid[row][c] == '0':
                continue
            else:
                for i in range(last_wall, row):
                    enemies_count[i][c] += curr
                last_wall = row + 1
                curr = 0

        for i in range(last_wall, row + 1):
            enemies_count[i][c] += curr

    return enemies_count


# -----------------------------------------------------------------------
"""
1218. Longest Arithmetic Subsequence of Given Difference

Given an integer array arr and an integer difference, return the length of the longest subsequence in arr which is an arithmetic sequence such that the difference between adjacent elements in the subsequence equals difference.

A subsequence is a sequence that can be derived from arr by deleting some or no elements without changing the order of the remaining elements.

 

Example 1:

Input: arr = [1,2,3,4], difference = 1
Output: 4
Explanation: The longest arithmetic subsequence is [1,2,3,4].
Example 2:

Input: arr = [1,3,5,7], difference = 1
Output: 1
Explanation: The longest arithmetic subsequence is any single element.
Example 3:

Input: arr = [1,5,7,8,5,3,4,2,1], difference = -2
Output: 4
Explanation: The longest arithmetic subsequence is [7,5,3,1].
"""


class Solution(object):
    def longestSubsequence(self, arr, difference):
        """
        :type arr: List[int]
        :type difference: int
        :rtype: int
        """

        diff = defaultdict(int)

        res = 0
        for val in arr:
            diff[val] = diff[val - difference] + 1

            res = max(res, diff[val])

        return res

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
334. Increasing Triplet Subsequence

Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.

 

Example 1:

Input: nums = [1,2,3,4,5]
Output: true
Explanation: Any triplet where i < j < k is valid.
Example 2:

Input: nums = [5,4,3,2,1]
Output: false
Explanation: No triplet exists.
Example 3:

Input: nums = [2,1,5,0,4,6]
Output: true
Explanation: The triplet (3, 4, 5) is valid because nums[3] == 0 < nums[4] == 4 < nums[5] == 6.
"""


class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        smallest, second_smallest = float('inf'), float('inf')

        for num in nums:
            if num <= smallest:
                smallest = num
            elif num <= second_smallest:
                second_smallest = num
            else:
                return True

        return False

    # time O(n)
    # space O(1)


"""

Aproach 1:

for each idx i,
    find the longest increasing subsequence from 0 to i, store the result in DP.
    if its larger than 3, return True

return False

# time O(n^2)
# space O(n)

 [2,1,5,0,4,6]
  1 1 2 1 2 3

Approach 2:

store the smallest, second smallest numbers
    find the number that is bigger than the second smallest


2 1 5 0 6
f = 0
s = 5

"""
# -----------------------------------------------------------------------
"""
91. Decode Ways

A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The answer is guaranteed to fit in a 32-bit integer.

 

Example 1:

Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
Example 2:

Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
Example 3:

Input: s = "0"
Output: 0
Explanation: There is no character that is mapped to a number starting with 0.
The only valid mappings with 0 are 'J' -> "10" and 'T' -> "20", neither of which start with 0.
Hence, there are no valid ways to decode this since all digits need to be mapped.
Example 4:

Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
"""


class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """

        return count_encode(s, 0, {})

    # time O(n)
    # space O(n)


def count_encode(s, idx, memo):
    if idx >= len(s):
        return 1

    if idx in memo:
        return memo[idx]

    curr_count = 0
    for end in range(idx + 1, len(s) + 1):

        curr_nums = s[idx:end]
        if int(curr_nums) > 26:
            break

        if is_valid(s[idx:end]):
            curr_count += count_encode(s, end, memo)

    memo[idx] = curr_count

    return memo[idx]


def is_valid(nums):
    if int(nums) < 1 or int(nums) > 26:
        return False

    if nums[0] == '0':
        return False

    return True


# -----------------------------------------------------------------------
"""
338. Counting Bits

Given an integer num, return an array of the number of 1's in the binary representation of every number in the range [0, num].

 

Example 1:

Input: num = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10
Example 2:

Input: num = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
 

Constraints:

0 <= num <= 105
 

Follow up:

It is very easy to come up with a solution with run time O(32n). Can you do it in linear time O(n) and possibly in a single pass?
Could you solve it in O(n) space complexity?
Can you do it without using any built-in function (i.e., like __builtin_popcount in C++)?
"""


class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """

        count = [0] * (num + 1)
        if num == 0:
            return count
        count[1] = 1

        sequence = 1
        for curr in range(2, num + 1):
            if is_power_2(curr):
                sequence *= 2
                count[curr] = 1
            else:
                count[curr] = count[curr - sequence] + 1

        return count

    # time O(n)
    # space O(n)


def is_power_2(x):
    return (x & (x - 1)) == 0 and x != 0


# -----------------------------------------------------------------------
"""
1182. Shortest Distance to Target Color

You are given an array colors, in which there are three colors: 1, 2 and 3.

You are also given some queries. Each query consists of two integers i and c, return the shortest distance between the given index i and the target color c. If there is no solution return -1.

 

Example 1:

Input: colors = [1,1,2,1,3,2,2,3,3], queries = [[1,3],[2,2],[6,1]]
Output: [3,0,3]
Explanation: 
The nearest 3 from index 1 is at index 4 (3 steps away).
The nearest 2 from index 2 is at index 2 itself (0 steps away).
The nearest 1 from index 6 is at index 3 (3 steps away).
Example 2:

Input: colors = [1,2], queries = [[0,3]]
Output: [-1]
Explanation: There is no 3 in the array.

"""


class Solution(object):
    def shortestDistanceColor(self, colors, queries):
        """
        :type colors: List[int]
        :type queries: List[List[int]]
        :rtype: List[int]
        """

        res = []
        color_to_idx = map_colors(colors)

        for idx, color in queries:
            res.append(get_shortest_distance(idx, color, colors, color_to_idx))

        return res

    # time O(n + q * log(n))
    # space O(n)


def map_colors(colors):
    idxs = defaultdict(list)

    for i, c in enumerate(colors):
        idxs[c].append(i)

    return idxs


def get_shortest_distance(idx, color, colors, color_to_idx):
    if idx < 0 or idx >= len(colors) or color not in color_to_idx:
        return -1

    curr_list = color_to_idx[color]

    place = bin_search(curr_list, idx)

    if place > 0:
        left = curr_list[place - 1]
    else:
        left = float('inf')

    if place < len(curr_list) - 1:
        right = curr_list[place + 1]
    else:
        right = float('inf')

    diff1 = abs(idx - left)
    diff2 = abs(idx - right)
    diff3 = abs(idx - curr_list[place])

    return min(diff1, diff2, diff3)


def bin_search(arr, num):
    left, right = 0, len(arr) - 1

    while left < right:
        mid = (left + right) >> 1

        if arr[mid] == num:
            return mid

        if num > arr[mid]:
            left = mid + 1
        else:
            right = mid - 1

    return left


# -----------------------------------------------------------------------
"""
379. Design Phone Directory

Design a phone directory that initially has maxNumbers empty slots that can store numbers. The directory should store numbers, check if a certain slot is empty or not, and empty a given slot.

Implement the PhoneDirectory class:

PhoneDirectory(int maxNumbers) Initializes the phone directory with the number of available slots maxNumbers.
int get() Provides a number that is not assigned to anyone.
bool check(int number) Returns true if the slot number is available and false otherwise.
void release(int number) Recycles or releases the slot number.
 

Example 1:

Input
["PhoneDirectory", "get", "get", "check", "get", "check", "release", "check"]
[[3], [], [], [2], [], [2], [2], [2]]
Output
[null, 0, 1, true, 2, false, null, true]

Explanation
PhoneDirectory phoneDirectory = new PhoneDirectory(3);
phoneDirectory.get();      // It can return any available phone number. Here we assume it returns 0.
phoneDirectory.get();      // Assume it returns 1.
phoneDirectory.check(2);   // The number 2 is available, so return true.
phoneDirectory.get();      // It returns 2, the only number that is left.
phoneDirectory.check(2);   // The number 2 is no longer available, so return false.
phoneDirectory.release(2); // Release number 2 back to the pool.
phoneDirectory.check(2);   // Number 2 is available again, return true.
"""


class PhoneDirectory(object):

    def __init__(self, maxNumbers):
        """
        Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory.
        :type maxNumbers: int
        """

        self.max_num = maxNumbers
        self.empty = [i for i in range(self.max_num)][::-1]
        self.seen = set()

    def get(self):
        """
        Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available.
        :rtype: int
        """

        if self.empty:
            empty = self.empty.pop()
            self.seen.add(empty)
            return empty

        return -1

    # time O(1)
    # space O(1)

    def check(self, number):
        """
        Check if a number is available or not.
        :type number: int
        :rtype: bool
        """

        return number not in self.seen

    # time O(1)
    # space O(1)

    def release(self, number):
        """
        Recycle or release a number.
        :type number: int
        :rtype: None
        """

        if number not in self.seen:
            return

        self.seen.remove(number)
        self.empty.append(number)

    # time O(1)
    # space O(1)


# Your PhoneDirectory object will be instantiated and called as such:
# obj = PhoneDirectory(maxNumbers)
# param_1 = obj.get()
# param_2 = obj.check(number)
# obj.release(number)

# -----------------------------------------------------------------------
"""
837. New 21 Game

Alice plays the following game, loosely based on the card game "21".

Alice starts with 0 points, and draws numbers while she has less than K points.  During each draw, she gains an integer number of points randomly from the range [1, W], where W is an integer.  Each draw is independent and the outcomes have equal probabilities.

Alice stops drawing numbers when she gets K or more points.  What is the probability that she has N or less points?

Example 1:

Input: N = 10, K = 1, W = 10
Output: 1.00000
Explanation:  Alice gets a single card, then stops.
Example 2:

Input: N = 6, K = 1, W = 10
Output: 0.60000
Explanation:  Alice gets a single card, then stops.
In 6 out of W = 10 possibilities, she is at or below N = 6 points.
Example 3:

Input: N = 21, K = 17, W = 10
Output: 0.73278
"""


class Solution(object):
    def new21Game(self, N, K, W):
        """
        :type N: int
        :type K: int
        :type W: int
        :rtype: float
        """

        if K == 0 or N > K - 1 + W:
            return 1

        p = [1.0] + [0.0] * N
        Wsum = 1.0
        for i in range(1, N + 1):
            p[i] = Wsum / W
            if i < K:
                Wsum += p[i]
            if i - W >= 0:
                Wsum -= p[i - W]

        return sum(p[K:])

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
388. Longest Absolute File Path

Suppose we have a file system that stores both files and directories. An example of one system is represented in the following picture:



Here, we have dir as the only directory in the root. dir contains two subdirectories, subdir1 and subdir2. subdir1 contains a file file1.ext and subdirectory subsubdir1. subdir2 contains a subdirectory subsubdir2, which contains a file file2.ext.

In text form, it looks like this (with ⟶ representing the tab character):

dir
⟶ subdir1
⟶ ⟶ file1.ext
⟶ ⟶ subsubdir1
⟶ subdir2
⟶ ⟶ subsubdir2
⟶ ⟶ ⟶ file2.ext
If we were to write this representation in code, it will look like this: "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext". Note that the '\n' and '\t' are the new-line and tab characters.

Every file and directory has a unique absolute path in the file system, which is the order of directories that must be opened to reach the file/directory itself, all concatenated by '/'s. Using the above example, the absolute path to file2.ext is "dir/subdir2/subsubdir2/file2.ext". Each directory name consists of letters, digits, and/or spaces. Each file name is of the form name.extension, where name and extension consist of letters, digits, and/or spaces.

Given a string input representing the file system in the explained format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return 0.

"""


class Solution(object):
    def lengthLongestPath(self, input):
        """
        :type input: str
        :rtype: int
        """

        dirs = input.split('\n')

        curr_count = 0

        path = [dirs[0]]
        max_length = 0
        curr_length = len(path[-1])

        for i in range(1, len(dirs)):
            if '.' in path[-1]:
                max_length = max(max_length, curr_length + len(path) - 1)

            count = dirs[i].count('\t')
            if count > curr_count:
                curr_count = count
                path.append(dirs[i].replace('\t', ''))
                curr_length += len(path[-1])

            elif count == curr_count:
                curr_length -= len(path[-1])
                path[-1] = dirs[i].replace('\t', '')
                curr_length += len(path[-1])
            else:
                diff = curr_count - count
                while diff >= 0:
                    curr_length -= len(path[-1])
                    path.pop()
                    diff -= 1
                path.append(dirs[i].replace('\t', ''))
                curr_length += len(path[-1])
                curr_count = count

        if '.' in path[-1]:
            max_length = max(max_length, curr_length + len(path) - 1)

        return max_length

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
1605. Find Valid Matrix Given Row and Column Sums

You are given two arrays rowSum and colSum of non-negative integers where rowSum[i] is the sum of the elements in the ith row and colSum[j] is the sum of the elements of the jth column of a 2D matrix. In other words, you do not know the elements of the matrix, but you do know the sums of each row and column.

Find any matrix of non-negative integers of size rowSum.length x colSum.length that satisfies the rowSum and colSum requirements.

Return a 2D array representing any matrix that fulfills the requirements. It's guaranteed that at least one matrix that fulfills the requirements exists.

 

Example 1:

Input: rowSum = [3,8], colSum = [4,7]
Output: [[3,0],
         [1,7]]
Explanation:
0th row: 3 + 0 = 3 == rowSum[0]
1st row: 1 + 7 = 8 == rowSum[1]
0th column: 3 + 1 = 4 == colSum[0]
1st column: 0 + 7 = 7 == colSum[1]
The row and column sums match, and all matrix elements are non-negative.
Another possible matrix is: [[1,2],
                             [3,5]]
Example 2:

Input: rowSum = [5,7,10], colSum = [8,6,8]
Output: [[0,5,0],
         [6,1,0],
         [2,0,8]]
Example 3:

Input: rowSum = [14,9], colSum = [6,9,8]
Output: [[0,9,5],
         [6,0,3]]
Example 4:

Input: rowSum = [1,0], colSum = [1]
Output: [[1],
         [0]]
Example 5:

Input: rowSum = [0], colSum = [0]
Output: [[0]]
"""


class Solution(object):
    def restoreMatrix(self, rowSum, colSum):
        """
        :type rowSum: List[int]
        :type colSum: List[int]
        :rtype: List[List[int]]
        """

        n = len(rowSum)
        m = len(colSum)

        result = [[0 for _ in range(m)] for _ in range(n)]

        i = 0
        j = 0
        while i < n and j < m:
            result[i][j] = min(rowSum[i], colSum[j])
            rowSum[i] -= result[i][j]
            colSum[j] -= result[i][j]
            if rowSum[i] == 0:
                i += 1
            if colSum[j] == 0:
                j += 1

        return result

    # time O(n + m)
    # space O(n*m)


# -----------------------------------------------------------------------
"""
95. Unique Binary Search Trees II

Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.

 

Example 1:


Input: n = 3
Output: [[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
Example 2:

Input: n = 1
Output: [[1]]
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """

        if n == 0:
            return []

        return generate_trees(1, n)


def generate_trees(start, end):
    if start > end:
        return [None, ]

    res = []

    for root in range(start, end + 1):

        left = generate_trees(start, root - 1)
        right = generate_trees(root + 1, end)

        for l in left:
            for r in right:
                current = TreeNode(root)
                current.left = l
                current.right = r
                res.append(current)

    return res


# -----------------------------------------------------------------------
"""
337. House Robber III

The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root.

Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

 

Example 1:


Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
Example 2:


Input: root = [3,4,5,1,3,null,1]
Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
 

Constraints:

The number of nodes in the tree is in the range [1, 104].
0 <= Node.val <= 104

"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        return max(max_rob_2(root))

        # time O(n)
        # space O(h)

        return max_rob(root, False)

    # time O(n^2)
    # space O(h)


def max_rob_2(node):
    if not node:
        return 0, 0

    take_left, leave_left = max_rob_2(node.left)
    take_right, leave_right = max_rob_2(node.right)

    return node.val + leave_left + leave_right, max(take_left, leave_left) + max(take_right, leave_right)


def max_rob(node, took_parent):
    if not node:
        return 0

    curr = 0
    if not took_parent:
        curr = max_rob(node.left, True) + max_rob(node.right, True) + node.val

    curr = max(curr, max_rob(node.left, False) + max_rob(node.right, False))

    return curr


"""

take_it, leave_it

left, right

1 = node.val + left[1] + right[1]

2 = max(left) + max(right)


return 1, 2

          7, 5  

    2, 3       3, 1     

        3, 0        1, 0


"""
# -----------------------------------------------------------------------
"""
686. Repeated String Match

Given two strings a and b, return the minimum number of times you should repeat string a so that string b is a substring of it. If it is impossible for b​​​​​​ to be a substring of a after repeating it, return -1.

Notice: string "abc" repeated 0 times is "",  repeated 1 time is "abc" and repeated 2 times is "abcabc".

 

Example 1:

Input: a = "abcd", b = "cdabcdab"
Output: 3
Explanation: We return 3 because by repeating a three times "abcdabcdabcd", b is a substring of it.
Example 2:

Input: a = "a", b = "aa"
Output: 2
Example 3:

Input: a = "a", b = "a"
Output: 1
Example 4:

Input: a = "abc", b = "wxyz"
Output: -1
"""


class Solution(object):
    def repeatedStringMatch(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """

        if b in a:
            return 1

        if set(a) != set(b):
            return -1

        min_rep = ((len(b) - len(a)) // len(a)) + 1

        curr = a * min_rep

        for _ in range(2):
            if b not in curr:
                min_rep += 1
                curr += a
            else:
                break

        return min_rep if b in curr else -1

    # time O(n + m)
    # space O(n + m)


"""
abcd

cdabcdab

8 - 4 = 4

4 // 1 => 1 + 1 = 2, 2 + 1


abcdabcdabcda


abcabcab

"""
# -----------------------------------------------------------------------
"""
288. Unique Word Abbreviation

The abbreviation of a word is a concatenation of its first letter, the number of characters between the first and last letter, and its last letter. If a word has only two characters, then it is an abbreviation of itself.

For example:

dog --> d1g because there is one letter between the first letter 'd' and the last letter 'g'.
internationalization --> i18n because there are 18 letters between the first letter 'i' and the last letter 'n'.
it --> it because any word with only two characters is an abbreviation of itself.
Implement the ValidWordAbbr class:

ValidWordAbbr(String[] dictionary) Initializes the object with a dictionary of words.
boolean isUnique(string word) Returns true if either of the following conditions are met (otherwise returns false):
There is no word in dictionary whose abbreviation is equal to word's abbreviation.
For any word in dictionary whose abbreviation is equal to word's abbreviation, that word and word are the same.
 

Example 1:

Input
["ValidWordAbbr", "isUnique", "isUnique", "isUnique", "isUnique"]
[[["deer", "door", "cake", "card"]], ["dear"], ["cart"], ["cane"], ["make"]]
Output
[null, false, true, false, true]

"""
from collections import defaultdict


class ValidWordAbbr(object):

    def __init__(self, dictionary):
        """
        :type dictionary: List[str]
        """

        self.word_to_abbreviation = defaultdict(str)
        self.abbreviation_to_word = defaultdict(set)

        for word in dictionary:
            curr = self.get_abbriviation(word)
            self.word_to_abbreviation[word] = curr
            self.abbreviation_to_word[curr].add(word)

    # time O(n * l)
    # space O(n * l)

    def get_abbriviation(self, word):

        if not word:
            return ''

        n = len(word)
        if n == 1 or n == 2:
            return word

        return word[0] + str(n - 2) + word[-1]

    # time O(1)
    # space O(1)

    def isUnique(self, word):
        """
        :type word: str
        :rtype: bool
        """

        if word in self.word_to_abbreviation:
            curr = self.word_to_abbreviation[word]
        else:
            curr = self.get_abbriviation(word)

        if curr not in self.abbreviation_to_word:
            return True

        if len(self.abbreviation_to_word[curr]) > 1 or word not in self.abbreviation_to_word[curr]:
            return False

        return True


# Your ValidWordAbbr object will be instantiated and called as such:
# obj = ValidWordAbbr(dictionary)
# param_1 = obj.isUnique(word)
# -----------------------------------------------------------------------

"""
271. Encode and Decode Strings

Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Machine 1 (sender) has the function:

string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}
Machine 2 (receiver) has the function:
vector<string> decode(string s) {
  //... your code
  return strs;
}
So Machine 1 does:

string encoded_string = encode(strs);
and Machine 2 does:

vector<string> strs2 = decode(encoded_string);
strs2 in Machine 2 should be the same as strs in Machine 1.

Implement the encode and decode methods.

 

Example 1:

Input: dummy_input = ["Hello","World"]
Output: ["Hello","World"]
Explanation:
Machine 1:
Codec encoder = new Codec();
String msg = encoder.encode(strs);
Machine 1 ---msg---> Machine 2

Machine 2:
Codec decoder = new Codec();
String[] strs = decoder.decode(msg);
Example 2:

Input: dummy_input = [""]
Output: [""]
 

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] contains any possible characters out of 256 valid ASCII characters.
 

Follow up:

Could you write a generalized algorithm to work on any possible characters?
Could you solve the problem without using any serialize methods (such as eval)?
"""


class Codec:

    def encode(self, strs):
        """Encodes a list of strings to a single string.

        :type strs: List[str]
        :rtype: str
        """

        d = unichr(257)

        return d.join(strs)

    # time O(n)
    # space O(1)

    def decode(self, s):
        """Decodes a single string to a list of strings.

        :type s: str
        :rtype: List[str]
        """

        d = unichr(257)

        return s.split(d)

    # time O(n)
    # space O(1)


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))

# -----------------------------------------------------------------------
"""
1087. Brace Expansion

You are given a string s representing a list of words. Each letter in the word has one or more options.

If there is one option, the letter is represented as is.
If there is more than one option, then curly braces delimit the options. For example, "{a,b,c}" represents options ["a", "b", "c"].
For example, if s = "a{b,c}", the first character is always 'a', but the second character can be 'b' or 'c'. The original list is ["ab", "ac"].

Return all words that can be formed in this manner, sorted in lexicographical order.

 

Example 1:

Input: s = "{a,b}c{d,e}f"
Output: ["acdf","acef","bcdf","bcef"]
Example 2:

Input: s = "abcd"
Output: ["abcd"]
"""


class Solution(object):
    def expand(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        options = get_options(s)  # n log(n)

        res = []

        get_all_words(options, 0, [], res)  # product of all options * n

        return res


def get_options(s):
    options = []
    left = None
    for right, c in enumerate(s):
        if c == '{':
            left = right + 1
        elif c == '}':
            options.append(sorted(s[left:right].replace(',', '')))
            left = None
        elif left == None:
            options.append(c)

    return options


def get_all_words(options, idx, curr, res):
    if idx == len(options):
        res.append(''.join(curr))
        return

    for c in options[idx]:
        get_all_words(options, idx + 1, curr + [c], res)

# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------

# -----------------------------------------------------------------------


# -----------------------------------------------------------------------

# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
