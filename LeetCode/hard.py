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
"""
995. Minimum Number of K Consecutive Bit Flips

In an array A containing only 0s and 1s, a K-bit flip consists of choosing a (contiguous) subarray of length K and simultaneously changing every 0 in the subarray to 1, and every 1 in the subarray to 0.

Return the minimum number of K-bit flips required so that there is no 0 in the array.  If it is not possible, return -1.

 

Example 1:

Input: A = [0,1,0], K = 1
Output: 2
Explanation: Flip A[0], then flip A[2].
Example 2:

Input: A = [1,1,0], K = 2
Output: -1
Explanation: No matter how we flip subarrays of size 2, we can't make the array become [1,1,1].
"""


class Solution(object):
    def minKBitFlips(self, A, k):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        n = len(A)
        is_flipped = [0] * n
        flipped = 0
        res = 0
        for i in range(n):
            if i >= k:
                flipped ^= is_flipped[i - k]

            if flipped == A[i]:
                if i + k > n:
                    return -1
                is_flipped[i] = 1
                flipped ^= 1
                res += 1

        return res

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
295. Find Median from Data Stream

The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
 

Example 1:

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
 

Constraints:

-105 <= num <= 105
There will be at least one element in the data structure before calling findMedian.
At most 5 * 104 calls will be made to addNum and findMedian.
"""

import heapq


class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """

        self.size = 0
        self.min_heap = []
        self.max_heap = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """

        self.size += 1

        heapq.heappush(self.max_heap, num * -1)
        heapq.heappush(self.min_heap, heapq.heappop(self.max_heap) * -1)

        if len(self.min_heap) > len(self.max_heap):
            curr_min = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -1 * curr_min)

    # time O(log(n))
    # space O(1)

    def findMedian(self):
        """
        :rtype: float
        """

        if self.size % 2 == 1:
            return self.max_heap[0] * -1

        left = self.max_heap[0] * -1
        right = self.min_heap[0]

        return float(left + right) / 2

    # time O(1)
    # space O(1)


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

# -----------------------------------------------------------------------
"""
85. Maximal Rectangle

Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

 

Example 1:


Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.
Example 2:

Input: matrix = []
Output: 0
Example 3:

Input: matrix = [["0"]]
Output: 0
Example 4:

Input: matrix = [["1"]]
Output: 1
"""


class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """

        n = len(matrix)
        if n == 0:
            return 0
        m = len(matrix[0])
        curr_row = [0 for _ in range(m)]
        res = float('-inf')
        for row in range(n):
            for col in range(m):
                if matrix[row][col] == '0':
                    curr_row[col] = 0
                else:
                    curr_row[col] += 1
            res = max(res, calc_area(curr_row))

        return res

    # time O(n * m)
    # space O(m)


def calc_area(heights):
    left = left_smaller(heights)
    right = right_smaller(heights)

    max_area = 0

    for i in range(len(heights)):
        curr_area = (right[i] - left[i] + 1) * heights[i]
        max_area = max(max_area, curr_area)

    return max_area


def left_smaller(heights):
    stack = []
    smaller = []

    for i in range(len(heights)):
        while stack and heights[i] <= heights[stack[-1]]:
            stack.pop()

        smaller.append(stack[-1] + 1 if stack else 0)
        stack.append(i)

    return smaller


def right_smaller(heights):
    stack = []
    smaller = []

    for i in range(len(heights) - 1, -1, -1):

        while stack and heights[i] <= heights[stack[-1]]:
            stack.pop()

        smaller.append(stack[-1] - 1 if stack else len(heights) - 1)
        stack.append(i)

    return smaller[::-1]


# -----------------------------------------------------------------------
"""
149. Max Points on a Line

Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.

 

Example 1:


Input: points = [[1,1],[2,2],[3,3]]
Output: 3
Example 2:


Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4
"""

from collections import defaultdict


class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """

        if len(points) == 0:
            return 0

        max_line = 1
        for i in range(len(points)):
            lines = defaultdict(int)
            for j in range(i):
                line = get_line(points[i], points[j])
                if line not in lines:
                    lines[line] = 2
                else:
                    lines[line] += 1

                max_line = max(max_line, lines[line])

        return max_line

    # time O(n^2)
    # space O(n)


def get_line(point1, point2):
    # slope = dy / dx
    # line = y - y0 = m * (x - x0)
    # => y = m*x - m*x0 + y0
    # => c = -m*x0 + y0

    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        slope = float('inf')
    else:
        slope = float(y1 - y2) / (x1 - x2)

    if slope == float('inf'):
        constant = x1
    else:
        constant = y1 - slope * x1

    return slope, constant


# -----------------------------------------------------------------------
"""
465. Optimal Account Balancing

A group of friends went on holiday and sometimes lent each other money. For example, Alice paid for Bill's lunch for $10. Then later Chris gave Alice $5 for a taxi ride. We can model each transaction as a tuple (x, y, z) which means person x gave person y $z. Assuming Alice, Bill, and Chris are person 0, 1, and 2 respectively (0, 1, 2 are the person's ID), the transactions can be represented as [[0, 1, 10], [2, 0, 5]].

Given a list of transactions between a group of people, return the minimum number of transactions required to settle the debt.

Note:

A transaction will be given as a tuple (x, y, z). Note that x ≠ y and z > 0.
Person's IDs may not be linear, e.g. we could have the persons 0, 1, 2 or we could also have the persons 0, 2, 6.
Example 1:

Input:
[[0,1,10], [2,0,5]]

Output:
2

Explanation:
Person #0 gave person #1 $10.
Person #2 gave person #0 $5.

Two transactions are needed. One way to settle the debt is person #1 pays person #0 and #2 $5 each.
Example 2:

Input:
[[0,1,10], [1,0,1], [1,2,5], [2,0,5]]

"""


class Solution(object):
    def minTransfers(self, transactions):
        """
        :type transactions: List[List[int]]
        :rtype: int
        """

        debt = defaultdict(int)

        for from_, to, amount in transactions:
            debt[from_] -= amount
            debt[to] += amount

        debts = debt.values()

        return min_transactions(debts, 0)

    # time O(2^n)
    # space O(n)


def min_transactions(debts, idx):
    while idx < len(debts) and debts[idx] == 0:
        idx += 1

    if idx >= len(debts):
        return 0

    curr = float('inf')

    for i in range(idx + 1, len(debts)):

        if debts[i] * debts[idx] < 0:  # different signs

            debts[i] += debts[idx]
            curr = min(curr, min_transactions(debts, idx + 1) + 1)
            debts[i] -= debts[idx]

            if debts[i] + debts[idx] == 0:
                break

    return curr if curr != float('inf') else 0


# -----------------------------------------------------------------------
"""
1153. String Transforms Into Another String

Given two strings str1 and str2 of the same length, determine whether you can transform str1 into str2 by doing zero or more conversions.

In one conversion you can convert all occurrences of one character in str1 to any other lowercase English character.

Return true if and only if you can transform str1 into str2.

 

Example 1:

Input: str1 = "aabcc", str2 = "ccdee"
Output: true
Explanation: Convert 'c' to 'e' then 'b' to 'd' then 'a' to 'c'. Note that the order of conversions matter.
Example 2:

Input: str1 = "leetcode", str2 = "codeleet"
Output: false
Explanation: There is no way to transform str1 to str2.
"""

from collections import defaultdict


class Solution(object):
    def canConvert(self, str1, str2):
        """
        :type str1: str
        :type str2: str
        :rtype: bool
        """

        if str1 == str2:
            return True

        if len(str1) != len(str2):
            return False

        if len(set(str2)) >= 26:
            return False

        chars1 = defaultdict()

        for c1, c2 in zip(str1, str2):

            if c1 in chars1 and chars1[c1] != c2:
                return False

            chars1[c1] = c2

        return len(set(str2)) < 26

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
871. Minimum Number of Refueling Stops

A car travels from a starting position to a destination which is target miles east of the starting position.

Along the way, there are gas stations.  Each station[i] represents a gas station that is station[i][0] miles east of the starting position, and has station[i][1] liters of gas.

The car starts with an infinite tank of gas, which initially has startFuel liters of fuel in it.  It uses 1 liter of gas per 1 mile that it drives.

When the car reaches a gas station, it may stop and refuel, transferring all the gas from the station into the car.

What is the least number of refueling stops the car must make in order to reach its destination?  If it cannot reach the destination, return -1.

Note that if the car reaches a gas station with 0 fuel left, the car can still refuel there.  If the car reaches the destination with 0 fuel left, it is still considered to have arrived.

 

Example 1:

Input: target = 1, startFuel = 1, stations = []
Output: 0
Explanation: We can reach the target without refueling.
Example 2:

Input: target = 100, startFuel = 1, stations = [[10,100]]
Output: -1
Explanation: We can't reach the target (or even the first gas station).
Example 3:

Input: target = 100, startFuel = 10, stations = [[10,60],[20,30],[30,30],[60,40]]
Output: 2
Explanation: 
We start with 10 liters of fuel.
We drive to position 10, expending 10 liters of fuel.  We refuel from 0 liters to 60 liters of gas.
Then, we drive from position 10 to position 60 (expending 50 liters of fuel),
and refuel from 10 liters to 50 liters of gas.  We then drive to and reach the target.
We made 2 refueling stops along the way, so we return 2.
"""


class Solution(object):
    def minRefuelStops(self, target, startFuel, stations):
        """
        :type target: int
        :type startFuel: int
        :type stations: List[List[int]]
        :rtype: int
        """

        max_heap = []

        stations.append((target, float('inf')))

        res = 0
        prev = 0
        tank = startFuel

        for location, fuel in stations:
            tank -= location - prev

            while max_heap and tank < 0:
                tank += -heapq.heappop(max_heap)
                res += 1

            if tank < 0:
                return -1

            heapq.heappush(max_heap, -fuel)
            prev = location

        return res

    # time O(n * log(n))
    # space O(n)


#         max_location = [0] * (len(stations) + 1)

#         max_location[0] = startFuel


#         for idx, (location, fuel) in enumerate(stations):

#             for prev in range(idx, -1, -1):
#                 if max_location[prev] >= location:
#                     max_location[prev + 1] = max(max_location[prev + 1], max_location[prev] + fuel)


#         for i, loc in enumerate(max_location):
#             if loc >= target:
#                 return i

#         return -1

#     # time O(n^2)
#     # space O(n)


# -----------------------------------------------------------------------
"""
727. Minimum Window Subsequence

Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.

If there is no such window in S that covers all characters in T, return the empty string "". If there are multiple such minimum-length windows, return the one with the left-most starting index.

Example 1:

Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"
Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.
"deb" is not a smaller window because the elements of T in the window must occur in order.


Note:

All the strings in the input will only contain lowercase letters.
The length of S will be in the range [1, 20000].
The length of T will be in the range [1, 100].

"""


class Solution(object):
    def minWindow(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        n = len(S)
        m = len(T)

        t_ptr = 0
        right = 0
        res_len = float('inf')
        res_left = 0

        while right < n:

            if res_len == m:
                break

            if t_ptr < m and S[right] == T[t_ptr]:
                t_ptr += 1
            right += 1

            if t_ptr == m:
                left = minimize_window(S, T, right)
                curr_len = right - left
                if curr_len < res_len:
                    res_len = curr_len
                    res_left = left
                t_ptr = 0
                right = left + 1

        return S[res_left: res_left + res_len] if res_len != float('inf') else ''

    # time O(s * t)
    # space O(1)


def minimize_window(S, T, right):
    left = right - 1

    t_p = len(T) - 1
    while left >= 0 and t_p >= 0:
        if S[left] == T[t_p]:
            t_p -= 1
        left -= 1

    return left + 1


"""
1. go over all substrings
2. check if t is a subsequence of the current substring.
    => O(s^2 * t)

1. maintain sliding window.
2. when having all characters of t, take the minimal window
    => O(s * t)
"""
# -----------------------------------------------------------------------
"""
471. Encode String with Shortest Length

Given a non-empty string, encode the string such that its encoded length is the shortest.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times.

Note:

k will be a positive integer.
If an encoding process does not make the string shorter, then do not encode it. If there are several solutions, return any of them.
 

Example 1:

Input: s = "aaa"
Output: "aaa"
Explanation: There is no way to encode it such that it is shorter than the input string, so we do not encode it.
Example 2:

Input: s = "aaaaa"
Output: "5[a]"
Explanation: "5[a]" is shorter than "aaaaa" by 1 character.
Example 3:

Input: s = "aaaaaaaaaa"
Output: "10[a]"
Explanation: "a9[a]" or "9[a]a" are also valid solutions, both of them have the same length = 5, which is the same as "10[a]".
Example 4:

Input: s = "aabcaabcd"
Output: "2[aabc]d"
Explanation: "aabc" occurs twice, so one answer can be "2[aabc]d".
Example 5:

Input: s = "abbbabbbcabbbabbbc"
Output: "2[2[abbb]c]"
Explanation: "abbbabbbc" occurs twice, but "abbbabbbc" can also be encoded to "2[abbb]c", so one answer can be "2[2[abbb]c]".
"""


class Solution(object):
    def encode(self, s):
        """
        :type s: str
        :rtype: str
        """

        n = len(s)
        dp = [['' for _ in range(n)] for _ in range(n)]

        for i in range(n - 1, -1, -1):
            for j in range(i, n):

                curr = s[i:j + 1]
                dp[i][j] = curr

                if j - i < 4:
                    continue

                for k in range(i, j):
                    if len(dp[i][k] + dp[k + 1][j]) < len(dp[i][j]):
                        dp[i][j] = dp[i][k] + dp[k + 1][j]

                for k in range(len(curr)):
                    pattern = curr[:k + 1]

                    if pattern and len(curr) % len(pattern) == 0 and curr.replace(pattern, '') == '':
                        encode = str(len(curr) // len(pattern)) + '[' + dp[i][i + k] + ']'

                        if len(encode) < len(dp[i][j]):
                            dp[i][j] = encode

        return dp[0][n - 1]

    # time O(n^4)
    # space O(n^2)


# -----------------------------------------------------------------------
"""
715. Range Module

A Range Module is a module that tracks ranges of numbers. Your task is to design and implement the following interfaces in an efficient manner.

addRange(int left, int right) Adds the half-open interval [left, right), tracking every real number in that interval. Adding an interval that partially overlaps with currently tracked numbers should add any numbers in the interval [left, right) that are not already tracked.
queryRange(int left, int right) Returns true if and only if every real number in the interval [left, right) is currently being tracked.
removeRange(int left, int right) Stops tracking every real number currently being tracked in the interval [left, right).
Example 1:
addRange(10, 20): null
removeRange(14, 16): null
queryRange(10, 14): true (Every number in [10, 14) is being tracked)
queryRange(13, 15): false (Numbers like 14, 14.03, 14.17 in [13, 15) are not being tracked)
queryRange(16, 17): true (The number 16 in [16, 17) is still being tracked, despite the remove operation)
"""


class RangeModule(object):

    def __init__(self):
        self.ranges = []

    def get_bounds(self, left, right):

        i, j = 0, len(self.ranges) - 1
        for d in (100, 10, 1):
            while i + d - 1 < len(self.ranges) and self.ranges[i + d - 1][1] < left:
                i += d
            while j >= d - 1 and self.ranges[j - d + 1][0] > right:
                j -= d
        return i, j

    def addRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: None
        """

        i, j = self.get_bounds(left, right)

        if i <= j:
            left = min(left, self.ranges[i][0])
            right = max(right, self.ranges[j][1])

        self.ranges[i:j + 1] = [(left, right)]

    # time O(O(n))
    # space O(1)

    def queryRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: bool
        """

        i, j = self.get_bounds(left, right)
        if i >= len(self.ranges):
            return False

        return self.ranges[i][0] <= left and right <= self.ranges[i][1]

    # time O(log(n))
    # space O(1)

    def removeRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: None
        """

        i, j = self.get_bounds(left, right)

        merge = []
        for k in range(i, j + 1):
            if self.ranges[k][0] < left:
                merge.append((self.ranges[k][0], left))

            if self.ranges[k][1] > right:
                merge.append((right, self.ranges[k][1]))

        self.ranges[i:j + 1] = merge

    # time O(n)
    # space O(1)


# Your RangeModule object will be instantiated and called as such:
# obj = RangeModule()
# obj.addRange(left,right)
# param_2 = obj.queryRange(left,right)
# obj.removeRange(left,right)
# -----------------------------------------------------------------------
"""
1095. Find in Mountain Array

(This problem is an interactive problem.)

You may recall that an array A is a mountain array if and only if:

A.length >= 3
There exists some i with 0 < i < A.length - 1 such that:
A[0] < A[1] < ... A[i-1] < A[i]
A[i] > A[i+1] > ... > A[A.length - 1]
Given a mountain array mountainArr, return the minimum index such that mountainArr.get(index) == target.  If such an index doesn't exist, return -1.

You can't access the mountain array directly.  You may only access the array using a MountainArray interface:

MountainArray.get(k) returns the element of the array at index k (0-indexed).
MountainArray.length() returns the length of the array.
Submissions making more than 100 calls to MountainArray.get will be judged Wrong Answer.  Also, any solutions that attempt to circumvent the judge will result in disqualification.

 

Example 1:

Input: array = [1,2,3,4,5,3,1], target = 3
Output: 2
Explanation: 3 exists in the array, at index=2 and index=5. Return the minimum index, which is 2.
Example 2:

Input: array = [0,1,2,4,2,1], target = 3
Output: -1
Explanation: 3 does not exist in the array, so we return -1.

"""


# """
# This is MountainArray's API interface.
# You should not implement it, or speculate about its implementation
# """
# class MountainArray(object):
#    def get(self, index):
#        """
#        :type index: int
#        :rtype int
#        """
#
#    def length(self):
#        """
#        :rtype int
#        """

class Solution(object):
    def findInMountainArray(self, target, mountain_arr):
        """
        :type target: integer
        :type mountain_arr: MountainArray
        :rtype: integer
        """

        mount_idx = find_mountain(mountain_arr)
        check = bin_search(mountain_arr, 0, mount_idx, target, compare_asc)
        if check != -1:
            return check

        return bin_search(mountain_arr, mount_idx + 1, mountain_arr.length() - 1, target, compare_desc)

    # time O(log(n))
    # space O(1)


def compare_asc(val1, val2):
    if val2 == val1:
        return 0
    elif val2 > val1:
        return 1
    return -1


def compare_desc(val1, val2):
    if val1 == val2:
        return 0
    elif val2 > val1:
        return -1
    return 1


def bin_search(arr, left, right, target, compare):
    while left <= right:
        mid = (left + right) >> 1

        mid_el = arr.get(mid)

        if compare(mid_el, target) == 0:
            return mid
        elif compare(mid_el, target) > 0:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def find_mountain(arr):
    left, right = 1, arr.length() - 2

    while left <= right:
        mid = (left + right) >> 1

        mid_el = arr.get(mid)
        next_mid = arr.get(mid + 1)
        prev_mid = arr.get(mid - 1)

        if mid_el > next_mid and mid_el > prev_mid:
            return mid
        elif mid_el > prev_mid:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# -----------------------------------------------------------------------
"""
1499. Max Value of Equation

Given an array points containing the coordinates of points on a 2D plane, sorted by the x-values, where points[i] = [xi, yi] such that xi < xj for all 1 <= i < j <= points.length. You are also given an integer k.

Find the maximum value of the equation yi + yj + |xi - xj| where |xi - xj| <= k and 1 <= i < j <= points.length. It is guaranteed that there exists at least one pair of points that satisfy the constraint |xi - xj| <= k.

 

Example 1:

Input: points = [[1,3],[2,0],[5,10],[6,-10]], k = 1
Output: 4
Explanation: The first two points satisfy the condition |xi - xj| <= 1 and if we calculate the equation we get 3 + 0 + |1 - 2| = 4. Third and fourth points also satisfy the condition and give a value of 10 + -10 + |5 - 6| = 1.
No other pairs satisfy the condition, so we return the max of 4 and 1.
Example 2:

Input: points = [[0,0],[3,0],[9,2]], k = 3
Output: 3
Explanation: Only the first two points have an absolute difference of 3 or less in the x-values, and give the value of 0 + 0 + |0 - 3| = 3.
 

Constraints:

2 <= points.length <= 10^5
points[i].length == 2
-10^8 <= points[i][0], points[i][1] <= 10^8
0 <= k <= 2 * 10^8
points[i][0] < points[j][0] for all 1 <= i < j <= points.length
xi form a strictly increasing sequence.
"""

import heapq
from collections import deque


class Solution(object):
    def findMaxValueOfEquation(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: int
        """

        # solution 3
        res = float('-inf')
        queue = deque()

        for x, y in points:
            while queue and queue[0][1] < x - k:
                queue.popleft()

            if queue:
                res = max(res, queue[0][0] + x + y)

            while queue and queue[-1][0] <= y - x:
                queue.pop()

            queue.append([y - x, x])

        return res

        # time O(n)
        # space O(n)

        # solution 2
        res = float('-inf')
        max_heap = []

        for xj, yj in points:

            while max_heap and max_heap[0][1] < xj - k:
                heapq.heappop(max_heap)

            if max_heap:
                res = max(res, (-max_heap[0][0] + xj + yj))

            heapq.heappush(max_heap, (xj - yj, xj))

        return res

        # time O(n * log(n))
        # space O(n)

        # solution 1
        n = len(points)

        res = float('-inf')
        for i in range(n):
            for j in range(i + 1, n):

                target = points[i][0] + k

                end = bin_search(points, target, i + 1, n - 1)
                if end == -1:
                    continue
                res = max(res, calc_result(points, i, end))

        return res

        # time O(n^2)
        # space O(1)


def bin_search(points, target, start, end):
    res = -1
    while start <= end:
        mid = (start + end) >> 1

        if points[mid][0] > target:
            end = mid - 1
        else:
            res = mid
            start = mid + 1

    return res


def calc_result(points, start, end):
    curr = points[start]
    res = float('-inf')
    for j in range(start + 1, end + 1):
        res = max(res, calc_equation(curr, points[j]))

    return res


def calc_equation(p1, p2):
    return (p1[1] - p1[0]) + (p2[0] + p2[1])


"""

** yi + yj + |xi - xj| , i < j
since xi < xj then the equation is -> yi + yj + xj - xi

-> (yi - xi) + (yj + xj)


** xj - xi <= k -> k + xi >= xj

(yi - xi, xi)


for each point i:
    check all previous points j, if xi + k >= xj then:
        res = max(res, equation(pi, pj))

O(n^2)



[[0,0],[3,0],[6,5],[7,7],[9,2]] k = 3

-3 + 11 = 9

-1 + 11 = 10
-1 + 14 = 13

"""
# -----------------------------------------------------------------------
"""
759. Employee Free Time

We are given a list schedule of employees, which represents the working time for each employee.

Each employee has a list of non-overlapping Intervals, and these intervals are in sorted order.

Return the list of finite intervals representing common, positive-length free time for all employees, also in sorted order.

(Even though we are representing Intervals in the form [x, y], the objects inside are Intervals, not lists or arrays. For example, schedule[0][0].start = 1, schedule[0][0].end = 2, and schedule[0][0][0] is not defined).  Also, we wouldn't include intervals like [5, 5] in our answer, as they have zero length.

 

Example 1:

Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]
Explanation: There are a total of three employees, and all common
free time intervals would be [-inf, 1], [3, 4], [10, inf].
We discard any intervals that contain inf as they aren't finite.
Example 2:

Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
Output: [[5,6],[7,9]]
 

Constraints:

1 <= schedule.length , schedule[i].length <= 50
0 <= schedule[i].start < schedule[i].end <= 10^8
"""


# Definition for an Interval.
class Interval(object):
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end


class Solution(object):
    def employeeFreeTime(self, schedule):
        """
        :type schedule: [[Interval]]
        :rtype: [Interval]
        """
        s = 0
        e = 1

        events = []

        for worker in schedule:
            for interval in worker:
                events.append((interval.start, s))
                events.append((interval.end, e))

        events.sort()

        res = []
        prev = None
        counter = 0

        for time, event in events:

            if counter == 0 and prev != None:
                res.append(Interval(prev, time))

            if event == s:
                counter += 1
            else:
                counter -= 1
            prev = time

        return res

    # time O(n * log(n))
    # space O(n)


# -----------------------------------------------------------------------
"""
1776. Car Fleet II

There are n cars traveling at different speeds in the same direction along a one-lane road. You are given an array cars of length n, where cars[i] = [positioni, speedi] represents:

positioni is the distance between the ith car and the beginning of the road in meters. It is guaranteed that positioni < positioni+1.
speedi is the initial speed of the ith car in meters per second.
For simplicity, cars can be considered as points moving along the number line. Two cars collide when they occupy the same position. Once a car collides with another car, they unite and form a single car fleet. The cars in the formed fleet will have the same position and the same speed, which is the initial speed of the slowest car in the fleet.

Return an array answer, where answer[i] is the time, in seconds, at which the ith car collides with the next car, or -1 if the car does not collide with the next car. Answers within 10-5 of the actual answers are accepted.

 

Example 1:

Input: cars = [[1,2],[2,1],[4,3],[7,2]]
Output: [1.00000,-1.00000,3.00000,-1.00000]
Explanation: After exactly one second, the first car will collide with the second car, and form a car fleet with speed 1 m/s. After exactly 3 seconds, the third car will collide with the fourth car, and form a car fleet with speed 2 m/s.
Example 2:

Input: cars = [[3,4],[5,4],[6,3],[9,1]]
Output: [2.00000,1.00000,1.50000,-1.00000]
"""


class Solution(object):
    def getCollisionTimes(self, cars):
        """
        :type cars: List[List[int]]
        :rtype: List[float]
        """
        if len(cars) == 0:
            return []

        n = len(cars)
        times = [-1] * n
        stack = [n - 1]

        for i in range(n - 2, -1, -1):

            curr_p, curr_s = cars[i]

            while stack:
                last_p, last_s = cars[stack[-1]]
                last_idx = stack[-1]
                if last_s >= curr_s:
                    stack.pop()
                elif times[last_idx] >= 0 and calc_time(cars[i], cars[last_idx]) >= times[last_idx]:
                    stack.pop()
                else:
                    break

            if stack:
                times[i] = calc_time(cars[i], cars[stack[-1]])

            stack.append(i)

        return times

    # time O(n)
    # space O(n)


def calc_time(car1, car2):
    """
    s = d / t => t = d / s
    t = delta d / delta s
    """

    distance_diff = car2[0] - car1[0]
    speed_diff = car1[1] - car2[1]

    return float(distance_diff) / speed_diff


# -----------------------------------------------------------------------
"""
218. The Skyline Problem

A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Given the locations and heights of all the buildings, return the skyline formed by these buildings collectively.

The geometric information of each building is given in the array buildings where buildings[i] = [lefti, righti, heighti]:

lefti is the x coordinate of the left edge of the ith building.
righti is the x coordinate of the right edge of the ith building.
heighti is the height of the ith building.
You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height 0.

The skyline should be represented as a list of "key points" sorted by their x-coordinate in the form [[x1,y1],[x2,y2],...]. Each key point is the left endpoint of some horizontal segment in the skyline except the last point in the list, which always has a y-coordinate 0 and is used to mark the skyline's termination where the rightmost building ends. Any ground between the leftmost and rightmost buildings should be part of the skyline's contour.

Note: There must be no consecutive horizontal lines of equal height in the output skyline. For instance, [...,[2 3],[4 5],[7 5],[11 5],[12 7],...] is not acceptable; the three lines of height 5 should be merged into one in the final output as such: [...,[2 3],[4 5],[12 7],...]

 

Example 1:


Input: buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
Output: [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
Explanation:
Figure A shows the buildings of the input.
Figure B shows the skyline formed by those buildings. The red points in figure B represent the key points in the output list.
Example 2:

Input: buildings = [[0,2,3],[2,5,3]]
Output: [[0,3],[5,0]]
"""


class Solution(object):
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """

        return get_skyline(buildings, 0, len(buildings) - 1)

        # time O(n * log(n))
        # space O(n)


def get_skyline(buildings, left, right):
    if left > right:
        return []

    if left == right:
        l, r, h = buildings[left]

        return [[l, h], [r, 0]]

    mid = (left + right) >> 1

    left_skyline = get_skyline(buildings, left, mid)
    right_skyline = get_skyline(buildings, mid + 1, right)

    return merge_skyline(left_skyline, right_skyline)


def merge_skyline(left, right):
    p1, p2 = 0, 0
    res = []

    left_y, right_y, curr_y = 0, 0, 0

    while p1 < len(left) and p2 < len(right):
        left_point, right_point = left[p1], right[p2]

        if left_point[0] < right_point[0]:
            x, left_y = left_point
            p1 += 1
        else:
            x, right_y = right_point
            p2 += 1

        max_y = max(left_y, right_y)

        if curr_y != max_y:
            add_to_res(x, max_y, res)
            curr_y = max_y

    if p1 < len(left):
        add_rest(res, p1, left, curr_y)

    if p2 < len(right):
        add_rest(res, p2, right, curr_y)

    return res


def add_rest(res, p, skyline, curr_y):
    while p < len(skyline):
        x, y = skyline[p]

        if y != curr_y:
            add_to_res(x, y, res)
            curr_y = y
        p += 1


def add_to_res(x, y, res):
    if not res or res[-1][0] != x:
        res.append([x, y])
    else:
        res[-1][1] = y


# -----------------------------------------------------------------------
"""
305. Number of Islands II

You are given an empty 2D binary grid grid of size m x n. The grid represents a map where 0's represent water and 1's represent land. Initially, all the cells of grid are water cells (i.e., all the cells are 0's).

We may perform an add land operation which turns the water at position into a land. You are given an array positions where positions[i] = [ri, ci] is the position (ri, ci) at which we should operate the ith operation.

Return an array of integers answer where answer[i] is the number of islands after turning the cell (ri, ci) into a land.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

 

Example 1:


Input: m = 3, n = 3, positions = [[0,0],[0,1],[1,2],[2,1]]
Output: [1,1,2,3]
Explanation:
Initially, the 2d grid is filled with water.
- Operation #1: addLand(0, 0) turns the water at grid[0][0] into a land. We have 1 island.
- Operation #2: addLand(0, 1) turns the water at grid[0][1] into a land. We still have 1 island.
- Operation #3: addLand(1, 2) turns the water at grid[1][2] into a land. We have 2 islands.
- Operation #4: addLand(2, 1) turns the water at grid[2][1] into a land. We have 3 islands.
Example 2:

Input: m = 1, n = 1, positions = [[0,0]]
Output: [1]
"""


class UnionFind:
    def __init__(self, size):
        self.parent = [-1] * size
        self.size = [0] * size
        self.count = 0

    def find(self, idx):

        # path compression
        if self.parent[idx] != idx:
            self.parent[idx] = self.find(self.parent[idx])

        return self.parent[idx]

    def is_valid(self, i):

        if i < 0 or i >= len(self.parent) or self.parent[i] == -1:
            return False

        return True

    def union(self, i1, i2):

        if not self.is_valid(i1) or not self.is_valid(i2):
            return

        root1, root2 = self.find(i1), self.find(i2)

        if root1 == root2:
            return

        if self.size[root1] < self.size[root2]:
            self.parent[root1] = root2
            self.size[root2] += self.size[root1]
        else:
            self.parent[root2] = root1
            self.size[root1] += self.size[root2]

        self.count -= 1

    def get_count(self):
        return self.count

    def set_parent(self, i):
        if self.parent[i] != -1:
            return

        self.parent[i] = i
        self.size[i] = 1
        self.count += 1


class Solution(object):
    def numIslands2(self, m, n, positions):
        """
        :type m: int
        :type n: int
        :type positions: List[List[int]]
        :rtype: List[int]
        """

        islands_count = []

        uf = UnionFind(n * m)

        for row, col in positions:

            idx = row * n + col

            uf.set_parent(idx)

            for n_row, n_col in get_neighbors(row, col):
                curr_idx = n_row * n + n_col

                if not is_valid(n_row, n_col, m, n):
                    continue
                uf.union(idx, curr_idx)

            islands_count.append(uf.get_count())

        return islands_count

    # time O(n * m + l)
    # space O(n * m)


def is_valid(row, col, m, n):
    if row < 0 or col < 0 or row >= m or col >= n:
        return False
    return True


def get_neighbors(row, col):
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    res = []

    for dx, dy in dirs:
        res.append((row + dx, col + dy))

    return res


"""
2
2
[[0,0],[1,1],[0,1]]



3
3
[[0,1],[1,2],[2,1],[1,0],[0,2],[0,0],[1,1]]

0 1 0 
1 0 1
0 1 0
"""

# -----------------------------------------------------------------------
"""
683. K Empty Slots

You have n bulbs in a row numbered from 1 to n. Initially, all the bulbs are turned off. We turn on exactly one bulb every day until all bulbs are on after n days.

You are given an array bulbs of length n where bulbs[i] = x means that on the (i+1)th day, we will turn on the bulb at position x where i is 0-indexed and x is 1-indexed.

Given an integer k, return the minimum day number such that there exists two turned on bulbs that have exactly k bulbs between them that are all turned off. If there isn't such day, return -1.

 

Example 1:

Input: bulbs = [1,3,2], k = 1
Output: 2
Explanation:
On the first day: bulbs[0] = 1, first bulb is turned on: [1,0,0]
On the second day: bulbs[1] = 3, third bulb is turned on: [1,0,1]
On the third day: bulbs[2] = 2, second bulb is turned on: [1,1,1]
We return 2 because on the second day, there were two on bulbs with one off bulb between them.
Example 2:

Input: bulbs = [1,2,3], k = 1
Output: -1
"""

from bisect import bisect, insort


class Solution(object):
    def kEmptySlots(self, bulbs, k):
        """
        :type bulbs: List[int]
        :type k: int
        :rtype: int
        """

        turned_on = []

        for i, bulb in enumerate(bulbs):

            idx = bisect.bisect(turned_on, bulb)

            for nei in range(-1, 1, 1):
                new_idx = nei + idx
                if new_idx >= 0 and new_idx < len(turned_on):
                    if abs(turned_on[new_idx] - bulb) - 1 == k:
                        return i + 1
            bisect.insort(turned_on, bulb)

        return -1

    # time O(n * log(n))
    # space O(n)


# -----------------------------------------------------------------------
"""
115. Distinct Subsequences

Given two strings s and t, return the number of distinct subsequences of s which equals t.

A string's subsequence is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the remaining characters' relative positions. (i.e., "ACE" is a subsequence of "ABCDE" while "AEC" is not).

It is guaranteed the answer fits on a 32-bit signed integer.

 

Example 1:

Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from S.
rabbbit
rabbbit
rabbbit
Example 2:

Input: s = "babgbag", t = "bag"
Output: 5
Explanation:
As shown below, there are 5 ways you can generate "bag" from S.
babgbag
babgbag
babgbag
babgbag
babgbag
"""


class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """

        return distinct_sub(s, t, 0, 0, {})

    # time O(n * m)
    # space O(n * m)


def distinct_sub(s, t, i, j, memo):
    if j == len(t):
        return 1

    if i == len(s):
        return 0

    if (i, j) in memo:
        return memo[(i, j)]

    curr = 0
    if s[i] != t[j]:
        curr = distinct_sub(s, t, i + 1, j, memo)
    else:
        curr = distinct_sub(s, t, i + 1, j + 1, memo) + distinct_sub(s, t, i + 1, j, memo)

    memo[(i, j)] = curr

    return curr


# -----------------------------------------------------------------------
"""
1032. Stream of Characters

Implement the StreamChecker class as follows:

StreamChecker(words): Constructor, init the data structure with the given words.
query(letter): returns true if and only if for some k >= 1, the last k characters queried (in order from oldest to newest, including this letter just queried) spell one of the words in the given list.
 

Example:

StreamChecker streamChecker = new StreamChecker(["cd","f","kl"]); // init the dictionary.
streamChecker.query('a');          // return false
streamChecker.query('b');          // return false
streamChecker.query('c');          // return false
streamChecker.query('d');          // return true, because 'cd' is in the wordlist
streamChecker.query('e');          // return false
streamChecker.query('f');          // return true, because 'f' is in the wordlist
streamChecker.query('g');          // return false
streamChecker.query('h');          // return false
streamChecker.query('i');          // return false
streamChecker.query('j');          // return false
streamChecker.query('k');          // return false
streamChecker.query('l');          // return true, because 'kl' is in the wordlist
 

Note:

1 <= words.length <= 2000
1 <= words[i].length <= 2000
Words will only consist of lowercase English letters.
Queries will only consist of lowercase English letters.
The number of queries is at most 40000.
"""

from collections import deque


class Trie:
    def __init__(self):
        self.root = {}

    def add(self, word):

        curr = self.root

        for c in word:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]

        curr['*'] = word

    def match(self, word):

        curr = self.root

        for c in word:
            if '*' in curr:
                return True

            if c not in curr:
                return False

            curr = curr[c]

        return '*' in curr


class StreamChecker(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """

        self.trie = Trie()
        for word in words:
            self.trie.add(word[::-1])

        self.input = deque()

    # time O(n * m)
    # space O(n * m)

    def query(self, letter):
        """
        :type letter: str
        :rtype: bool
        """

        self.input.appendleft(letter)

        return self.trie.match(self.input)

    # time O(n)
    # space O(1)


# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)

# -----------------------------------------------------------------------
"""
1231. Divide Chocolate

You have one chocolate bar that consists of some chunks. Each chunk has its own sweetness given by the array sweetness.

You want to share the chocolate with your K friends so you start cutting the chocolate bar into K+1 pieces using K cuts, each piece consists of some consecutive chunks.

Being generous, you will eat the piece with the minimum total sweetness and give the other pieces to your friends.

Find the maximum total sweetness of the piece you can get by cutting the chocolate bar optimally.

 

Example 1:

Input: sweetness = [1,2,3,4,5,6,7,8,9], K = 5
Output: 6
Explanation: You can divide the chocolate to [1,2,3], [4,5], [6], [7], [8], [9]
Example 2:

Input: sweetness = [5,6,7,8,9,1,2,3,4], K = 8
Output: 1
Explanation: There is only one way to cut the bar into 9 pieces.
Example 3:

Input: sweetness = [1,2,2,1,2,2,1,2,2], K = 2
Output: 5
Explanation: You can divide the chocolate to [1,2,2], [1,2,2], [1,2,2]
 

Constraints:

0 <= K < sweetness.length <= 10^4
1 <= sweetness[i] <= 10^5
"""


class Solution(object):
    def maximizeSweetness(self, sweetness, K):
        """
        :type sweetness: List[int]
        :type K: int
        :rtype: int
        """

        left = min(sweetness)

        right = sum(sweetness)

        res = 0

        while left <= right:
            mid = (left + right) >> 1

            if can_cut(sweetness, mid, K):
                res = mid
                left = mid + 1
            else:
                right = mid - 1

        return res

    # time O(log(sum(arr)) * n)
    # space O(1)


def can_cut(sweetness, max_sweetness, k):
    curr = 0
    count = 0

    for val in sweetness:
        curr += val

        if curr >= max_sweetness:
            count += 1
            curr = 0

            if count > k:
                return True

    return count > k


# -----------------------------------------------------------------------
"""
568. Maximum Vacation Days

LeetCode wants to give one of its best employees the option to travel among N cities to collect algorithm problems. But all work and no play makes Jack a dull boy, you could take vacations in some particular cities and weeks. Your job is to schedule the traveling to maximize the number of vacation days you could take, but there are certain rules and restrictions you need to follow.

Rules and restrictions:
You can only travel among N cities, represented by indexes from 0 to N-1. Initially, you are in the city indexed 0 on Monday.
The cities are connected by flights. The flights are represented as a N*N matrix (not necessary symmetrical), called flights representing the airline status from the city i to the city j. If there is no flight from the city i to the city j, flights[i][j] = 0; Otherwise, flights[i][j] = 1. Also, flights[i][i] = 0 for all i.
You totally have K weeks (each week has 7 days) to travel. You can only take flights at most once per day and can only take flights on each week's Monday morning. Since flight time is so short, we don't consider the impact of flight time.
For each city, you can only have restricted vacation days in different weeks, given an N*K matrix called days representing this relationship. For the value of days[i][j], it represents the maximum days you could take vacation in the city i in the week j.
You're given the flights matrix and days matrix, and you need to output the maximum vacation days you could take during K weeks.

Example 1:
Input:flights = [[0,1,1],[1,0,1],[1,1,0]], days = [[1,3,1],[6,0,3],[3,3,3]]
Output: 12
Explanation: 
Ans = 6 + 3 + 3 = 12. 

One of the best strategies is:
1st week : fly from city 0 to city 1 on Monday, and play 6 days and work 1 day. 
(Although you start at city 0, we could also fly to and start at other cities since it is Monday.) 
2nd week : fly from city 1 to city 2 on Monday, and play 3 days and work 4 days.
3rd week : stay at city 2, and play 3 days and work 4 days.
Example 2:
Input:flights = [[0,0,0],[0,0,0],[0,0,0]], days = [[1,1,1],[7,7,7],[7,7,7]]
Output: 3
Explanation: 
Ans = 1 + 1 + 1 = 3. 

Since there is no flights enable you to move to another city, you have to stay at city 0 for the whole 3 weeks. 
For each week, you only have one day to play and six days to work. 
So the maximum number of vacation days is 3.
Example 3:
Input:flights = [[0,1,1],[1,0,1],[1,1,0]], days = [[7,0,0],[0,7,0],[0,0,7]]
Output: 21
Explanation:
Ans = 7 + 7 + 7 = 21

One of the best strategies is:
1st week : stay at city 0, and play 7 days. 
2nd week : fly from city 0 to city 1 on Monday, and play 7 days.
3rd week : fly from city 1 to city 2 on Monday, and play 7 days.
Note:
N and K are positive integers, which are in the range of [1, 100].
In the matrix flights, all the values are integers in the range of [0, 1].
In the matrix days, all the values are integers in the range [0, 7].
You could stay at a city beyond the number of vacation days, but you should work on the extra days, which won't be counted as vacation days.
If you fly from the city A to the city B and take the vacation on that day, the deduction towards vacation days will count towards the vacation days of city B in that week.
We don't consider the impact of flight hours towards the calculation of vacation days.
"""


class Solution(object):
    def maxVacationDays(self, flights, days):
        """
        :type flights: List[List[int]]
        :type days: List[List[int]]
        :rtype: int
        """

        return dfs(flights, days, 0, 0, {})
    # time O(n^2 * k)
    # space O(n * k)


def dfs(flights, days, curr_city, week_num, memo):
    if week_num == len(days[0]):
        return 0

    if (curr_city, week_num) in memo:
        return memo[(curr_city, week_num)]

    max_vac = 0

    for adj in range(len(flights)):
        if flights[curr_city][adj] == 1 or adj == curr_city:
            curr_vac = dfs(flights, days, adj, week_num + 1, memo) + days[adj][week_num]
            max_vac = max(max_vac, curr_vac)

    memo[(curr_city, week_num)] = max_vac
    return max_vac


# -----------------------------------------------------------------------
"""
1463. Cherry Pickup II

Given a rows x cols matrix grid representing a field of cherries. Each cell in grid represents the number of cherries that you can collect.

You have two robots that can collect cherries for you, Robot #1 is located at the top-left corner (0,0) , and Robot #2 is located at the top-right corner (0, cols-1) of the grid.

Return the maximum number of cherries collection using both robots  by following the rules below:

From a cell (i,j), robots can move to cell (i+1, j-1) , (i+1, j) or (i+1, j+1).
When any robot is passing through a cell, It picks it up all cherries, and the cell becomes an empty cell (0).
When both robots stay on the same cell, only one of them takes the cherries.
Both robots cannot move outside of the grid at any moment.
Both robots should reach the bottom row in the grid.
 

Example 1:



Input: grid = [[3,1,1],[2,5,1],[1,5,5],[2,1,1]]
Output: 24
Explanation: Path of robot #1 and #2 are described in color green and blue respectively.
Cherries taken by Robot #1, (3 + 2 + 5 + 2) = 12.
Cherries taken by Robot #2, (1 + 5 + 5 + 1) = 12.
Total of cherries: 12 + 12 = 24.
Example 2:



Input: grid = [[1,0,0,0,0,0,1],[2,0,0,0,0,3,0],[2,0,9,0,0,0,0],[0,3,0,5,4,0,0],[1,0,2,3,0,0,6]]
Output: 28
Explanation: Path of robot #1 and #2 are described in color green and blue respectively.
Cherries taken by Robot #1, (1 + 9 + 5 + 2) = 17.
Cherries taken by Robot #2, (1 + 3 + 4 + 3) = 11.
Total of cherries: 17 + 11 = 28.
Example 3:

Input: grid = [[1,0,0,3],[0,0,0,3],[0,0,3,3],[9,0,3,3]]
Output: 22
Example 4:

Input: grid = [[1,1],[1,1]]
Output: 4
 

Constraints:

rows == grid.length
cols == grid[i].length
2 <= rows, cols <= 70
"""


class Solution(object):
    def cherryPickup(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        return max_cherries(grid, (0, 0), (0, len(grid[0]) - 1), {}, set())

    # time O(n * m^2)
    # space O(n * m)


def max_cherries(grid, robot1, robot2, memo, visited):
    n = len(grid)
    if robot1[0] == n - 1 and robot2[0] == n - 1:
        if robot1[1] != robot2[1]:
            return grid[robot1[0]][robot1[1]] + grid[robot2[0]][robot2[1]]
        return grid[robot1[0]][robot1[1]]

    if (robot1, robot2) in memo:
        return memo[(robot1, robot2)]

    curr = 0
    for adj1 in get_neighbors(robot1):
        if not is_valid(grid, adj1):
            continue
        if adj1 in visited:
            continue

        visited.add(adj1)

        temp1 = grid[adj1[0]][adj1[1]]
        grid[adj1[0]][adj1[1]] = 0
        for adj2 in get_neighbors(robot2):
            if not is_valid(grid, adj2):
                continue

            if adj2 in visited:
                continue

            visited.add(adj2)

            temp2 = grid[adj2[0]][adj2[1]]
            grid[adj2[0]][adj2[1]] = 0

            take = max_cherries(grid, adj1, adj2, memo, visited) + temp1 + temp2
            if take != float('-inf'):
                curr = max(curr, take)
            grid[adj2[0]][adj2[1]] = temp2
            visited.remove(adj2)

        grid[adj1[0]][adj1[1]] = temp1
        visited.remove(adj1)

    memo[(robot1, robot2)] = curr

    return curr


def get_neighbors(point):
    row, col = point

    dirs = [[0, 0], [1, 0], [1, -1], [1, 1]]

    res = []
    for dx, dy in dirs:
        res.append((row + dx, col + dy))

    return res


def is_valid(grid, point):
    row, col = point

    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
        return False

    return True

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
