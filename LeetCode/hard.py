# -----------------------------------------------------------------------
"""
42. Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it can trap after raining.
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1].
In this case, 6 units of rain water (blue section) are being trapped.
"""


class Solution:

    #     def trap(self, height):
    #         if len(height) == 0:
    #             return 0
    #         n = len(height)
    #         leftMax, rightMax = [0] * n , [0] * n
    #         i, j = 0, len(height) - 1
    #         currLeftMax = height[0]
    #         currRightMax = height[len(height) - 1]
    #         while i < len(height):
    #             if height[i] > currLeftMax:
    #                 currLeftMax = height[i]

    #             if height[j] > currRightMax:
    #                 currRightMax = height[j]

    #             leftMax[i] = currLeftMax
    #             rightMax[j] = currRightMax
    #             i += 1
    #             j -= 1
    #         ans = 0
    #         for i in range(n):
    #             ans += (min(leftMax[i], rightMax[i]) - height[i])

    #         return ans

    #         # time O(n)
    #         # space O(n)

    def trap(self, height):

        left, right, lMax, rMax = 0, len(height) - 1, 0, 0
        ans = 0
        while left < right:
            if height[left] < height[right]:
                if height[left] >= lMax:
                    lMax = height[left]
                else:
                    ans += lMax - height[left]
                left += 1
            else:
                if height[right] >= rMax:
                    rMax = height[right]
                else:
                    ans += rMax - height[right]
                right -= 1

        return ans

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
297. Serialize and Deserialize Binary Tree

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        if not root:
            return ''

        res = []
        q = deque([root])

        while len(q) > 0:

            curr = q.popleft()

            if curr:
                q.append(curr.left)
                q.append(curr.right)
                res.append(str(curr.val))
            else:
                res.append('#')

        return ' '.join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        if not data:
            return None

        data = data.split(' ')
        res = TreeNode(data[0])

        curr_level = deque([res])
        idx = 1

        while idx < len(data) and curr_level:

            curr = curr_level.popleft()

            if data[idx] != '#':
                curr.left = TreeNode(data[idx])
                curr_level.append(curr.left)

            idx += 1
            if data[idx] != '#':
                curr.right = TreeNode(data[idx])
                curr_level.append(curr.right)
            idx += 1

        return res


# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
# -----------------------------------------------------------------------
"""
72. Edit Distance

Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character
 

Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
"""


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        dists = [[j + i for j in range(m + 1)] for i in range(n + 1)]
        for row in range(1, n + 1):
            for col in range(1, m + 1):
                if word1[row - 1] == word2[col - 1]:
                    dists[row][col] = dists[row - 1][col - 1]
                else:

                    dists[row][col] = min(dists[row - 1][col - 1], min(dists[row - 1][col], dists[row][col - 1])) + 1

        return dists[n][m]

    # time O(n * m)
    # space O(n * m) "can be reduced to O(min(m, n))"


# -----------------------------------------------------------------------
"""
128. Longest Consecutive Sequence

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

 
Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4

Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
 

Constraints:

0 <= nums.length <= 104
-109 <= nums[i] <= 109
"""


class Solution:
    def longestConsecutive(self, nums):
        # approach 1:
        # 1. sort the arr
        # 2. take the longest sequence
        # time O(n*log(n))
        # space O(1)

        # approach 2: using DFS
        nums = set(nums)

        longest_seq = 0
        visited = set()
        for num in nums:
            length = visit_sequence(nums, num, visited)
            longest_seq = max(longest_seq, length)

        return longest_seq

    # time O(n)
    # space O(n)


def visit_sequence(nums, num, visited):
    if num in visited:
        return 1

    visited.add(num)

    counter = 1
    right_num = num + 1
    while right_num in nums:
        visited.add(right_num)
        counter += 1
        right_num += 1

    left_num = num - 1
    while left_num in nums:
        visited.add(left_num)
        counter += 1
        left_num -= 1

    return counter


# -----------------------------------------------------------------------
"""
489. Robot Room Cleaner

Given a robot cleaner in a room modeled as a grid.

Each cell in the grid can be empty or blocked.

The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 90 degrees.

When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it stays on the current cell.

Design an algorithm to clean the entire room using only the 4 given APIs shown below.

interface Robot {
  // returns true if next cell is open and robot moves into the cell.
  // returns false if next cell is obstacle and robot stays on the current cell.
  boolean move();

  // Robot will stay on the same cell after calling turnLeft/turnRight.
  // Each turn will be 90 degrees.
  void turnLeft();
  void turnRight();

  // Clean the current cell.
  void clean();
}
Example:

Input:
room = [
  [1,1,1,1,1,0,1,1],
  [1,1,1,1,1,0,1,1],
  [1,0,1,1,1,1,1,1],
  [0,0,0,1,0,0,0,0],
  [1,1,1,1,1,1,1,1]
],
row = 1,
col = 3

Explanation:
All grids in the room are marked by either 0 or 1.
0 means the cell is blocked, while 1 means the cell is accessible.
The robot initially starts at the position of row=1, col=3.
From the top left corner, its position is one row below and three columns right.
"""


# """
# This is the robot's control interface.
# You should not implement it, or speculate about its implementation
# """
# class Robot:
#    def move(self):
#        """
#        Returns true if the cell in front is open and robot moves into the cell.
#        Returns false if the cell in front is blocked and robot stays in the current cell.
#        :rtype bool
#        """
#
#    def turnLeft(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def turnRight(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def clean(self):
#        """
#        Clean the current cell.
#        :rtype void
#        """

class Solution:
    def cleanRoom(self, robot):
        """
        :type robot: Robot
        :rtype: None
        """

        visited = set()

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        def go_back():
            robot.turnRight()
            robot.turnRight()
            robot.move()
            robot.turnRight()
            robot.turnRight()

        def dfs_visit(location, direction):

            visited.add(location)
            robot.clean()

            for i in range(4):
                new_direction = (direction + i) % 4

                new_location = (location[0] + directions[new_direction][0], \
                                location[1] + directions[new_direction][1])

                if new_location not in visited and robot.move():
                    dfs_visit(new_location, new_direction)
                    go_back()
                robot.turnRight()

        dfs_visit((0, 0), 0)


# -----------------------------------------------------------------------
"""
76. Minimum Window Substring

Given two strings s and t, return the minimum window in s which will contain all the characters in t. If there is no such window in s that covers all characters in t, return the empty string "".

Note that If there is such a window, it is guaranteed that there will always be only one unique minimum window in s.

 

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Example 2:

Input: s = "a", t = "a"
Output: "a"
 
"""

from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        start_idx, length = 0, float('inf')
        char_counts = Counter(t)
        left = 0
        remained_chars = len(char_counts)
        for i, char in enumerate(s):
            # we need to process only chars that are in the t string
            if char not in char_counts:
                continue

            char_counts[char] -= 1
            # if the counter reached to 0, it means we have encountered all the needed counter for thi char
            if char_counts[char] == 0:
                remained_chars -= 1

            # encountered all the chars with all the needed counter
            if remained_chars == 0:
                # try to minimize the window
                while left <= i and remained_chars == 0:
                    if s[left] in char_counts:
                        char_counts[s[left]] += 1
                        if char_counts[s[left]] > 0:
                            remained_chars += 1
                    left += 1
                # check if its less than the current min window
                curr_length = i - (left - 1) + 1
                if curr_length < length:
                    length = curr_length
                    start_idx = left - 1

        return '' if length == float('inf') else s[start_idx: start_idx + length]

    # time O(n + m)
    # space O(m)


# -----------------------------------------------------------------------
"""
23. Merge k Sorted Lists

You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

 

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
Example 2:

Input: lists = []
Output: []
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


import heapq


class Solution:
    def mergeKLists(self, lists):

        min_heap = []
        for idx, head in enumerate(lists):
            if head:
                heapq.heappush(min_heap, (head.val, idx, head))

        res = ListNode('DUMMY')
        curr = res
        while len(min_heap) > 0:

            min_val, idx, min_node = heapq.heappop(min_heap)
            curr.next = min_node
            curr = curr.next
            min_node = min_node.next
            if min_node:
                heapq.heappush(min_heap, (min_node.val, idx, min_node))

        return res.next

    # time O(n * log(k))
    # space O(k)


# -----------------------------------------------------------------------
"""
124. Binary Tree Maximum Path Sum

A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any path.

 

Example 1:


Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
Example 2:


Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        return get_max_path(root)[1]

    # time O(n)
    # space O(h)


def get_max_path(node):
    if not node:
        return 0, float('-inf')  # max path , max path sum

    left = get_max_path(node.left)
    right = get_max_path(node.right)

    max_path = max(max(left[0], right[0]) + node.val, node.val)

    max_path_sum = max(node.val, left[1], right[1], left[0] + node.val, right[0] + node.val,
                       left[0] + right[0] + node.val)

    return max_path, max_path_sum


# -----------------------------------------------------------------------
"""
127. Word Ladder

A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words such that:

The first word in the sequence is beginWord.
The last word in the sequence is endWord.
Only one letter is different between each adjacent pair of words in the sequence.
Every word in the sequence is in wordList.
Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

 

Example 1:

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog" with 5 words.
Example 2:

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: The endWord "cog" is not in wordList, therefore there is no possible transformation.
"""

from collections import defaultdict, deque


class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        n = len(wordList)
        indexes = get_indexes_mapping(wordList)
        if beginWord not in wordList:
            wordList.append(beginWord)
            indexes[beginWord] = n
            n += 1
        if endWord not in wordList:
            return 0

        graph = generate_graph(wordList, indexes)
        start_idx = indexes[beginWord]
        end_idx = indexes[endWord]

        q = deque([[start_idx, 1]])
        visited = set([start_idx])

        while len(q) > 0:
            curr_word, curr_dist = q.popleft()

            for adj in graph[curr_word]:
                if adj not in visited:
                    q.append([adj, curr_dist + 1])
                    visited.add(adj)
                    if adj == end_idx:
                        return curr_dist + 1
        return 0

    # time O(n * L^2)
    # space O(n * L^2)


def get_indexes_mapping(words):
    indexes = {}
    for i, val in enumerate(words):
        indexes[val] = i

    return indexes


def generate_graph(wordList, indexes):
    graph = defaultdict(set)
    for i, word in enumerate(wordList):
        curr_word = list(word)
        for pos in range(len(curr_word)):
            temp = curr_word[pos]
            for dx in range(0, 26):
                curr_word[pos] = chr(dx + 97)
                temp_word = ''.join(curr_word)
                if temp_word == word:
                    continue
                if temp_word in indexes:
                    graph[i].add(indexes[temp_word])
                    graph[indexes[temp_word]].add(i)
            curr_word[pos] = temp

    return graph


# -----------------------------------------------------------------------
"""
329. Longest Increasing Path in a Matrix

Given an m x n integers matrix, return the length of the longest increasing path in matrix.

From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

 

Example 1:


Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
Output: 4
Explanation: The longest increasing path is [1, 2, 6, 9].
Example 2:


Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
Output: 4
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
Example 3:

Input: matrix = [[1]]
Output: 1
"""

from collections import defaultdict

directions = {'U': [-1, 0], 'R': [0, 1], 'D': [1, 0], 'L': [0, -1]}


class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        n = len(matrix)
        m = len(matrix[0])

        distances = defaultdict()
        max_dist = 0

        for row in range(n):
            for col in range(m):
                max_dist = max(max_dist, get_distance(matrix, distances, row, col))

        return max_dist

    # time O(n * m)
    # space O(n * m)


def is_valid(matrix, row, col):
    if row < 0 or col < 0 or row >= len(matrix) or col >= len(matrix[0]):
        return False

    return True


def get_neighbors(row, col):
    res = []
    for dir_ in directions:
        dx = directions[dir_][0]
        dy = directions[dir_][1]
        res.append((row + dx, col + dy))

    return res


def get_distance(matrix, distances, row, col):
    if (row, col) in distances:
        return distances[(row, col)]

    curr_dist = 0

    for n_row, n_col in get_neighbors(row, col):
        if is_valid(matrix, n_row, n_col) and matrix[n_row][n_col] > matrix[row][col]:
            curr_dist = max(curr_dist, get_distance(matrix, distances, n_row, n_col))

    curr_dist += 1
    distances[(row, col)] = curr_dist

    return curr_dist


# -----------------------------------------------------------------------
"""
4. Median of Two Sorted Arrays

Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

 

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
Example 3:

Input: nums1 = [0,0], nums2 = [0,0]
Output: 0.00000
Example 4:

Input: nums1 = [], nums2 = [1]
Output: 1.00000
Example 5:

Input: nums1 = [2], nums2 = []
Output: 2.00000
"""


class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """

        if len(nums2) < len(nums1):
            return self.findMedianSortedArrays(nums2, nums1)

        n = len(nums1)
        m = len(nums2)

        if n == 0:
            return median(nums2)
        if m == 0:
            return median(nums1)

        smaller, larger = nums1, nums2

        left, right = 0, n

        while left <= right:
            mid = (left + right) // 2
            small_part = mid
            large_part = ((n + m + 1) // 2) - small_part

            left_small = float('-inf') if small_part == 0 else smaller[small_part - 1]
            right_small = float('inf') if small_part == n else smaller[small_part]

            left_large = float('-inf') if large_part == 0 else larger[large_part - 1]
            right_large = float('inf') if large_part == m else larger[large_part]

            if left_small <= right_large and left_large <= right_small:
                if (n + m) % 2 == 0:
                    left_val = max(left_small, left_large)
                    right_val = min(right_small, right_large)

                    return (left_val + right_val) / 2.0
                else:
                    return max(left_small, left_large)

            if left_large > right_small:
                left = mid + 1
            else:
                right = mid - 1

        # time O(log(min(n, m)))
        # space O(1)


def median(nums):
    n = len(nums)
    mid = n // 2
    if n % 2 == 1:
        return nums[mid]

    return (nums[mid] + nums[mid - 1]) / 2.0


# -----------------------------------------------------------------------
"""
315. Count of Smaller Numbers After Self

You are given an integer array nums and you have to return a new counts array. The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

 

Example 1:

Input: nums = [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
"""


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left_count = 0
        self.right = self.left = None


class AVL:
    def __init__(self):
        self.root = None

    def insert(self, val):
        self.root = self.insert_val(self.root, val)

    def insert_val(self, node, val):
        if not node:
            return TreeNode(val)

        if val < node.val:
            node.left_count += 1
            node.left = self.insert_val(node.left, val)
        else:
            node.right = self.insert_val(node.right, val)

        return node

    def get_smaller_count(self, val):

        return self.get_smaller(self.root, val)

    def get_smaller(self, node, val):

        if node.val == val:
            return node.left_count

        if val < node.val:
            return self.get_smaller(node.left, val)
        else:
            return node.left_count + 1 + self.get_smaller(node.right, val)


class Solution(object):
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        res = []
        tree = AVL()
        for val in nums[::-1]:
            tree.insert(val)
            res.append(tree.get_smaller_count(val))

        return res[::-1]

    # time O(n * h)
    # space O(n)


# -----------------------------------------------------------------------
"""
425. Word Squares

Given a set of words (without duplicates), find all word squares you can build from them.

A sequence of words forms a valid word square if the kth row and column read the exact same string, where 0 ≤ k < max(numRows, numColumns).

For example, the word sequence ["ball","area","lead","lady"] forms a word square because each word reads the same both horizontally and vertically.

b a l l
a r e a
l e a d
l a d y
Note:
There are at least 1 and at most 1000 words.
All words will have the exact same length.
Word length is at least 1 and at most 5.
Each word contains only lowercase English alphabet a-z.
"""

from collections import defaultdict


class Solution(object):
    def wordSquares(self, words):
        """
        :type words: List[str]
        :rtype: List[List[str]]
        """

        res = []
        prefixes = build_prefixes(words)

        for word in words:
            build_square([word], 1, res, words, prefixes)

        return res

    # time O(n * n^l)
    # space O(n ^ l^2)


def build_prefixes(words):
    prefixes = defaultdict(list)

    for word in words:
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            prefixes[prefix].append(word)

    return prefixes


def build_square(curr, idx, res, words, prefixes):
    n = len(words[0])
    if idx == n:
        res.append(curr[:])
        return

    pref = ''.join([val[idx] for val in curr])

    for candidate in prefixes[pref]:
        build_square(curr + [candidate], idx + 1, res, words, prefixes)


# -----------------------------------------------------------------------
"""
212. Word Search II

Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

 

Example 1:


Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
Example 2:


Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []
 

Constraints:

m == board.length
n == board[i].length
1 <= m, n <= 12
board[i][j] is a lowercase English letter.
1 <= words.length <= 3 * 104
1 <= words[i].length <= 10
words[i] consists of lowercase English letters.
All the strings of words are unique.
"""

from collections import defaultdict


class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """

        prefix = starts_with_same(words)
        res = set()

        for row in range(len(board)):
            for col in range(len(board[0])):
                curr_char = board[row][col]
                if curr_char not in prefix:
                    continue

                for word in prefix[curr_char]:
                    if word in res:
                        continue

                    if can_generate(word, board, row, col, set(), 1):
                        res.add(word)

                if len(res) == len(words):
                    return res

        return list(res)

    # time O(n * m + K * l) n: rows, m: cols, k: num of words, l: largest word
    # space O(K * l)


def is_valid(row, col, board):
    if row < 0 or col < 0 or row >= len(board) or col >= len(board[0]):
        return False

    return True


def can_generate(word, board, row, col, seen, idx):
    if idx == len(word):
        return True

    if (row, col) in seen:
        return False

    seen.add((row, col))

    for n_row, n_col in get_neighbors(row, col):
        if (n_row, n_col) in seen or not is_valid(n_row, n_col, board):
            continue

        if board[n_row][n_col] == word[idx]:
            if can_generate(word, board, n_row, n_col, seen, idx + 1):
                return True

    return False


directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_neighbors(row, col):
    res = []
    for dx, dy in directions:
        res.append((row + dx, col + dy))

    return res


def starts_with_same(words):
    prefix = defaultdict(list)

    for word in words:
        prefix[word[0]].append(word)

    return prefix


# -----------------------------------------------------------------------
"""
857. Minimum Cost to Hire K Workers

There are N workers.  The i-th worker has a quality[i] and a minimum wage expectation wage[i].

Now we want to hire exactly K workers to form a paid group.  When hiring a group of K workers, we must pay them according to the following rules:

Every worker in the paid group should be paid in the ratio of their quality compared to other workers in the paid group.
Every worker in the paid group must be paid at least their minimum wage expectation.
Return the least amount of money needed to form a paid group satisfying the above conditions.

 

Example 1:

Input: quality = [10,20,5], wage = [70,50,30], K = 2
Output: 105.00000
Explanation: We pay 70 to 0-th worker and 35 to 2-th worker.
Example 2:

Input: quality = [3,1,10,10,1], wage = [4,8,2,2,7], K = 3
Output: 30.66667
Explanation: We pay 4 to 0-th worker, 13.33333 to 2-th and 3-th workers seperately. 
"""

import heapq


class Worker:
    def __init__(self, quality, wage):
        self.quality = quality
        self.wage = wage
        self.min_ratio = wage / quality


class Solution(object):
    def mincostToHireWorkers(self, quality, wage, k):
        """
        :type quality: List[int]
        :type wage: List[int]
        :type K: int
        :rtype: float
        """

        workers = []
        for q, w in zip(quality, wage):
            workers.append(Worker(q, w))

        workers.sort(key=lambda x: x.min_ratio)
        min_wages = float('inf')
        qualities = 0
        max_heap = []

        for w in workers:
            qualities += w.quality
            heapq.heappush(max_heap, -1 * w.quality)

            if len(max_heap) > k:
                max_q = heapq.heappop(max_heap) * -1
                qualities -= max_q

            if len(max_heap) == k:
                min_wages = min(min_wages, w.min_ratio * qualities)

        return min_wages

    # time O(n * log(n))
    # space O(n)


# -----------------------------------------------------------------------
"""
642. Design Search Autocomplete System

Design a search autocomplete system for a search engine. Users may input a sentence (at least one word and end with a special character '#'). For each character they type except '#', you need to return the top 3 historical hot sentences that have prefix the same as the part of sentence already typed. Here are the specific rules:

The hot degree for a sentence is defined as the number of times a user typed the exactly same sentence before.
The returned top 3 hot sentences should be sorted by hot degree (The first is the hottest one). If several sentences have the same degree of hot, you need to use ASCII-code order (smaller one appears first).
If less than 3 hot sentences exist, then just return as many as you can.
When the input is a special character, it means the sentence ends, and in this case, you need to return an empty list.
Your job is to implement the following functions:

The constructor function:

AutocompleteSystem(String[] sentences, int[] times): This is the constructor. The input is historical data. Sentences is a string array consists of previously typed sentences. Times is the corresponding times a sentence has been typed. Your system should record these historical data.

Now, the user wants to input a new sentence. The following function will provide the next character the user types:

List<String> input(char c): The input c is the next character typed by the user. The character will only be lower-case letters ('a' to 'z'), blank space (' ') or a special character ('#'). Also, the previously typed sentence should be recorded in your system. The output will be the top 3 historical hot sentences that have prefix the same as the part of sentence already typed.

 
Example:
Operation: AutocompleteSystem(["i love you", "island","ironman", "i love leetcode"], [5,3,2,2])
The system have already tracked down the following sentences and their corresponding times:
"i love you" : 5 times
"island" : 3 times
"ironman" : 2 times
"i love leetcode" : 2 times
Now, the user begins another search:

Operation: input('i')
Output: ["i love you", "island","i love leetcode"]
Explanation:
There are four sentences that have prefix "i". Among them, "ironman" and "i love leetcode" have same hot degree. Since ' ' has ASCII code 32 and 'r' has ASCII code 114, "i love leetcode" should be in front of "ironman". Also we only need to output top 3 hot sentences, so "ironman" will be ignored.

Operation: input(' ')
Output: ["i love you","i love leetcode"]
Explanation:
There are only two sentences that have prefix "i ".

Operation: input('a')
Output: []
Explanation:
There are no sentences that have prefix "i a".

Operation: input('#')
Output: []
Explanation:
The user finished the input, the sentence "i a" should be saved as a historical sentence in system. And the following input will be counted as a new search.

 
Note:

The input sentence will always start with a letter and end with '#', and only one blank space will exist between two words.
The number of complete sentences that to be searched won't exceed 100. The length of each sentence including those in the historical data won't exceed 100.
Please use double-quote instead of single-quote when you write test cases even for a character input.
Please remember to RESET your class variables declared in class AutocompleteSystem, as static/class variables are persisted across multiple test cases. Please see here for more details.
"""


class Trie:
    def __init__(self):
        self.root = {}

    def add(self, word, times):

        curr = self.root

        for c in word:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]

        if '*' in curr:
            curr['*'][1] += times
        else:
            curr['*'] = [word, times]

    def starts_with(self, pref):

        if pref == '':
            return []

        curr = self.root

        for c in pref:
            if c not in curr:
                return []
            curr = curr[c]

        res = []
        self.get_words(curr, res)

        return res

    def get_words(self, curr, res):

        for c in curr:
            if c == '*':
                res.append(curr[c])
            else:
                self.get_words(curr[c], res)


class AutocompleteSystem(object):

    def __init__(self, sentences, times):
        """
        :type sentences: List[str]
        :type times: List[int]
        """

        self.curr_input = []
        self.k = 3
        self.trie = Trie()
        for i, sen in enumerate(sentences):
            self.trie.add(sen, times[i])

        # time O(n * l)
        # space O(n * l)

    def input(self, c):
        """
        :type c: str
        :rtype: List[str]
        """

        if c == '#':
            # add the word to trie
            self.trie.add(''.join(self.curr_input), 1)
            self.curr_input = []
            return []

        self.curr_input.append(c)

        words = self.trie.starts_with(''.join(self.curr_input))

        words.sort()
        words.sort(key=lambda x: x[1], reverse=True)

        return [words[i][0] for i in range(min(self.k, len(words)))]

    # time O(l + m + m * log(m))
    # space O(m)


# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)

# -----------------------------------------------------------------------
"""
126. Word Ladder II

A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words such that:

The first word in the sequence is beginWord.
The last word in the sequence is endWord.
Only one letter is different between each adjacent pair of words in the sequence.
Every word in the sequence is in wordList.
Given two words, beginWord and endWord, and a dictionary wordList, return all the shortest transformation sequences from beginWord to endWord, or an empty list if no such sequence exists.

 

Example 1:

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
Example 2:

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: []
Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.
 

Constraints:

1 <= beginWord.length <= 10
endWord.length == beginWord.length
1 <= wordList.length <= 5000
wordList[i].length == beginWord.length
beginWord, endWord, and wordList[i] consist of lowercase English letters.
beginWord != endWord
All the strings in wordList are unique.
"""

from collections import defaultdict, deque


class Solution(object):
    def findLadders(self, start, end, words):
        if end not in words:
            return []
        graph = generate_graph(start, end, words)
        return graph.get_paths(start, end)

    # time O(n * L)
    # space O(n * L)


class Graph:
    def __init__(self):
        self.edges = defaultdict(set)
        self.vertices = set()

    def add(self, u, v):
        self.edges[u].add(v)
        self.edges[v].add(u)
        self.vertices.add(u)
        self.vertices.add(v)

    def get_paths(self, start, end):

        paths = []

        q = deque([(start, [])])
        visited = set()

        min_len = float('inf')

        while q:
            curr, path = q.popleft()

            if len(path) > min_len:
                continue

            path.append(curr)

            if curr == end:
                paths.append(path)
                min_len = min(min_len, len(path))
            else:
                for neigh in self.edges[curr]:
                    if neigh in visited:
                        continue
                    q.append((neigh, [v for v in path]))

            visited.add(curr)

        return [path for path in paths if len(path) == min_len]


def generate_graph(start, end, words):
    words = set(words)

    idx_to_chars = generate_idx_to_char(words)

    graph = Graph()

    add_word_to_graph(graph, start, words, idx_to_chars)
    add_word_to_graph(graph, end, words, idx_to_chars)

    for word in words:
        add_word_to_graph(graph, word, words, idx_to_chars)

    return graph


def add_word_to_graph(graph, word, words, idx_to_chars):
    list_word = list(word)
    for i in range(len(word)):
        for c in idx_to_chars[i]:
            new_word = ''.join(list_word[:i] + [c] + list_word[i + 1:])
            if new_word in words and new_word != word:
                graph.add(word, new_word)


# for each idx, save all possible chars
def generate_idx_to_char(words):
    idx_to_chars = defaultdict(set)

    for word in words:
        for i, c in enumerate(word):
            idx_to_chars[i].add(c)

    return idx_to_chars


# -----------------------------------------------------------------------
"""
123. Best Time to Buy and Sell Stock III

You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

 

Example 1:

Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.
Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
Example 4:

Input: prices = [1]
Output: 0
"""


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        n = len(prices)

        t1_cost, t2_cost = float('inf'), float('inf')
        t1_profit, t2_profit = 0, 0

        for price in prices:
            t1_cost = min(t1_cost, price)
            t1_profit = max(t1_profit, price - t1_cost)

            t2_cost = min(t2_cost, price - t1_profit)
            t2_profit = max(t2_profit, price - t2_cost)

        return t2_profit

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
1406. Stone Game III

Alice and Bob continue their games with piles of stones. There are several stones arranged in a row, and each stone has an associated value which is an integer given in the array stoneValue.

Alice and Bob take turns, with Alice starting first. On each player's turn, that player can take 1, 2 or 3 stones from the first remaining stones in the row.

The score of each player is the sum of values of the stones taken. The score of each player is 0 initially.

The objective of the game is to end with the highest score, and the winner is the player with the highest score and there could be a tie. The game continues until all the stones have been taken.

Assume Alice and Bob play optimally.

Return "Alice" if Alice will win, "Bob" if Bob will win or "Tie" if they end the game with the same score.

 

Example 1:

Input: values = [1,2,3,7]
Output: "Bob"
Explanation: Alice will always lose. Her best move will be to take three piles and the score become 6. Now the score of Bob is 7 and Bob wins.
Example 2:

Input: values = [1,2,3,-9]
Output: "Alice"
Explanation: Alice must choose all the three piles at the first move to win and leave Bob with negative score.
If Alice chooses one pile her score will be 1 and the next move Bob's score becomes 5. The next move Alice will take the pile with value = -9 and lose.
If Alice chooses two piles her score will be 3 and the next move Bob's score becomes 3. The next move Alice will take the pile with value = -9 and also lose.
Remember that both play optimally so here Alice will choose the scenario that makes her win.
Example 3:

Input: values = [1,2,3,6]
Output: "Tie"
Explanation: Alice cannot win this game. She can end the game in a draw if she decided to choose all the first three piles, otherwise she will lose.
Example 4:

Input: values = [1,2,3,-1,-2,-3,7]
Output: "Alice"
Example 5:

Input: values = [-1,-2,-3]
Output: "Tie"
"""


class Solution(object):
    def stoneGameIII(self, stones):
        n = len(stones)
        scores = [0] * n

        for i in range(n - 1, -1, -1):
            taken = 0
            scores[i] = float('-inf')

            for k in range(3):
                if i + k >= n:
                    break
                taken += stones[i + k]
                next_val = 0 if i + k + 1 >= n else scores[i + k + 1]
                scores[i] = max(scores[i], taken - next_val)

        if scores[0] > 0:
            return 'Alice'
        elif scores[0] < 0:
            return 'Bob'

        return 'Tie'

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
410. Split Array Largest Sum

Given an array nums which consists of non-negative integers and an integer m, you can split the array into m non-empty continuous subarrays.

Write an algorithm to minimize the largest sum among these m subarrays.

 

Example 1:

Input: nums = [7,2,5,10,8], m = 2
Output: 18
Explanation:
There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.
Example 2:

Input: nums = [1,2,3,4,5], m = 2
Output: 9
Example 3:

Input: nums = [1,4,4], m = 3
Output: 4
"""


class Solution(object):
    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """

        left = max(nums)
        right = sum(nums)
        res = float('inf')

        while left <= right:
            mid = (left + right) >> 1
            count = count_sub_arrays(nums, mid)

            if count <= m:
                res = min(res, mid)
                right = mid - 1
            else:
                left = mid + 1

        return res

    # time O(n * log(sum(nums)))
    # space O(1)


def count_sub_arrays(nums, max_sum):
    count = 0
    curr_sum = 0
    for num in nums:
        curr_sum += num
        if curr_sum > max_sum:
            count += 1
            curr_sum = num

    return count + 1


# -----------------------------------------------------------------------
"""
135. Candy

There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
Return the minimum number of candies you need to have to distribute the candies to the children.

 

Example 1:

Input: ratings = [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
Example 2:

Input: ratings = [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
The third child gets 1 candy because it satisfies the above two conditions.
"""


class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """

        n = len(ratings)
        candies = [1] * n

        for i in range(1, n):

            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1

        final_res = candies[-1]

        for i in range(n - 2, -1, -1):

            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)

            final_res += candies[i]

        return final_res

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
843. Guess the Word

This problem is an interactive problem new to the LeetCode platform.

We are given a word list of unique words, each word is 6 letters long, and one word in this list is chosen as secret.

You may call master.guess(word) to guess a word.  The guessed word should have type string and must be from the original list with 6 lowercase letters.

This function returns an integer type, representing the number of exact matches (value and position) of your guess to the secret word.  Also, if your guess is not in the given wordlist, it will return -1 instead.

For each test case, you have 10 guesses to guess the word. At the end of any number of calls, if you have made 10 or less calls to master.guess and at least one of these guesses was the secret, you pass the testcase.

Besides the example test case below, there will be 5 additional test cases, each with 100 words in the word list.  The letters of each word in those testcases were chosen independently at random from 'a' to 'z', such that every word in the given word lists is unique.

Example 1:
Input: secret = "acckzz", wordlist = ["acckzz","ccbazz","eiowzz","abcczz"]

Explanation:

master.guess("aaaaaa") returns -1, because "aaaaaa" is not in wordlist.
master.guess("acckzz") returns 6, because "acckzz" is secret and has all 6 matches.
master.guess("ccbazz") returns 3, because "ccbazz" has 3 matches.
master.guess("eiowzz") returns 2, because "eiowzz" has 2 matches.
master.guess("abcczz") returns 4, because "abcczz" has 4 matches.

We made 5 calls to master.guess and one of them was the secret, so we pass the test case.
"""

# """
# This is Master's API interface.
# You should not implement it, or speculate about its implementation
# """
# class Master(object):
#    def guess(self, word):
#        """
#        :type word: str
#        :rtype int
#        """

from random import shuffle, randint


class Solution(object):
    def findSecretWord(self, wordlist, master):
        """
        :type wordlist: List[Str]
        :type master: Master
        :rtype: None
        """

        shuffle(wordlist)

        for _ in range(10):
            guess = get_random_choice(wordlist)
            matched = master.guess(guess)

            wordlist = [word for word in wordlist if match(word, guess) == matched]

        # time O(n)
        # space O(n)


def get_random_choice(words):
    i = randint(0, len(words) - 1)

    return words[i]


def match(word1, word2):
    return sum(c1 == c2 for c1, c2 in zip(word1, word2))


# -----------------------------------------------------------------------
"""
1074. Number of Submatrices That Sum to Target

Given a matrix and a target, return the number of non-empty submatrices that sum to target.

A submatrix x1, y1, x2, y2 is the set of all cells matrix[x][y] with x1 <= x <= x2 and y1 <= y <= y2.

Two submatrices (x1, y1, x2, y2) and (x1', y1', x2', y2') are different if they have some coordinate that is different: for example, if x1 != x1'.

 

Example 1:


Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
Output: 4
Explanation: The four 1x1 submatrices that only contain 0.
Example 2:

Input: matrix = [[1,-1],[-1,1]], target = 0
Output: 5
Explanation: The two 1x2 submatrices, plus the two 2x1 submatrices, plus the 2x2 submatrix.
Example 3:

Input: matrix = [[904]], target = 0
Output: 0
 

Constraints:

1 <= matrix.length <= 100
1 <= matrix[0].length <= 100
-1000 <= matrix[i] <= 1000
-10^8 <= target <= 10^8
"""

from collections import defaultdict


class Solution(object):
    def numSubmatrixSumTarget(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: int
        """

        n = len(matrix)
        m = len(matrix[0])

        pre_sum = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

        for row in range(1, n + 1):
            for col in range(1, m + 1):
                top = pre_sum[row - 1][col]
                left = pre_sum[row][col - 1]
                diagonal = pre_sum[row - 1][col - 1]
                pre_sum[row][col] = top + left - diagonal + matrix[row - 1][col - 1]

        count = 0
        for row in range(1, n + 1):

            for curr_row in range(row, n + 1):

                h_map = defaultdict(int)
                h_map[0] = 1

                for col in range(1, m + 1):
                    curr_sum = pre_sum[curr_row][col] - pre_sum[row - 1][col]

                    count += h_map[curr_sum - target]

                    h_map[curr_sum] += 1

        return count

    # time O(n^2 * m)
    # space O(n * m)

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
