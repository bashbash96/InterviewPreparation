import collections

# -----------------------------------------------------------------------
"""
173. Binary Search Tree Iterator

Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

Example:

BSTIterator iterator = new BSTIterator(root);
iterator.next();    // return 3
iterator.next();    // return 7
iterator.hasNext(); // return true
iterator.next();    // return 9
iterator.hasNext(); // return true
iterator.next();    // return 15
iterator.hasNext(); // return true
iterator.next();    // return 20
iterator.hasNext(); // return false

Note:

next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
You may assume that next() call will always be valid, that is, there will be at least a next smallest number
in the BST when next() is called.
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator:

    def __init__(self, root):
        self.stack = []
        self.getMostLeft(root)

        # time O(h)
        # space O(h)

    def getMostLeft(self, node):
        while node:
            self.stack.append(node)
            node = node.left

        # time O(h)
        # space O(h)

    def next(self) -> int:
        """
        @return the next smallest number
        """
        if self.hasNext():
            curr = self.stack.pop()
            data = curr.val
            self.getMostLeft(curr.right)
            return data
        return -1

        # time O(1)
        # space O(h)

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return len(self.stack) > 0

        # time O(1)
        # space O(1)


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()

# -----------------------------------------------------------------------
"""
Find longest subarray with absolute diff between any two elements less than K

Example: [2, 5, 10, 7, 3, 9, 0, 6, 11, 17], K = 6
answer = [5, 10, 7]

"""


def longestSubArray(nums, k):
    minQ = collections.deque([])
    maxQ = collections.deque([])

    maxLen = 0
    res = []
    left = 0

    for i, num in enumerate(nums):
        while minQ and num <= nums[minQ[-1]]:
            minQ.pop()
        while maxQ and num >= nums[maxQ[-1]]:
            maxQ.pop()

        minQ.append(i)
        maxQ.append(i)

        while minQ and maxQ and nums[maxQ[0]] - nums[minQ[0]] > k:

            left += 1
            if minQ[0] < left:
                minQ.popleft()
            if maxQ[0] < left:
                maxQ.popleft()

        if maxLen < i - left + 1:
            maxLen = i - left + 1
            res = nums[left:i + 1]

    return res

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
239. Sliding Window Maximum

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very 
right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. 
Return the max sliding window.

Follow up:
Could you solve it in linear time?

Example:

Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3
Output: [3,3,5,5,6,7] 
Explanation: 

Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 
"""


class Solution:
    def maxSlidingWindow(self, nums, k):
        maxQ = collections.deque([])

        for i in range(k):
            while maxQ and nums[i] >= nums[maxQ[-1]]:
                maxQ.pop()
            maxQ.append(i)

        res = [nums[maxQ[0]]]

        for i in range(k, len(nums)):
            while maxQ and maxQ[0] < (i - k) + 1:
                maxQ.popleft()

            while maxQ and nums[i] >= nums[maxQ[-1]]:
                maxQ.pop()
            maxQ.append(i)
            res.append(nums[maxQ[0]])

        return res

        # time O(n)
        # space O(n)


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
 
Constraints:

1 <= arr.length <= 300
1 <= arr[0].length <= 300
0 <= arr[i][j] <= 1
"""


class Solution:
    def countSquares(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])

        res = 0
        for row in range(rows):
            for col in range(cols):
                if matrix[row][col] == 1:
                    res += 1
                elif matrix[row][col] == 0:
                    continue
                if row == 0 or col == 0:
                    continue

                minSquares = min(matrix[row - 1][col - 1], min(matrix[row - 1][col], matrix[row][col - 1]))
                res += minSquares
                matrix[row][col] += minSquares

        return res

        # time O(m*n)
        # space O(1)


# -----------------------------------------------------------------------
"""
Design a Data structure to store and retrieve contacts.
eg: michael, michelle are the names stored, when the user enters 'mich' it should return all the contacts that match 
this prefix. If nothing is entered, it should bring all the contacts from the dictionary

"""


class Contacts:
    def __init__(self):
        self.root = {}
        self.contacts = set()

    def store(self, contact):
        if contact in self.contacts:
            return
        self.contacts.add(contact)
        curr = self.root
        for char in contact:
            if char not in curr:
                curr[char] = {}
            curr = curr[char]
        curr['*'] = '*'

        # time O(n)
        # space O(n)

    def retrieve(self, prefix):
        if prefix == '':
            return self.contacts
        res = []
        curr = self.root
        for char in prefix:
            if char not in curr:
                return []
            curr = curr[char]
        prefix = [char for char in prefix]
        self.getContactsWithPrefix(curr, prefix, res)

        return res

        # time O(l^2) -> l : the longest word with prefix , n :
        # space O(l^2)

    def getContactsWithPrefix(self, node, prefix, res):
        if node == '*' or not node:
            res.append(''.join(prefix[:-1]))
            return
        for char in node:
            prefix.append(char)
            self.getContactsWithPrefix(node[char], prefix, res)
            prefix.pop()


# -----------------------------------------------------------------------

"""
211. Add and Search Word - Data structure design

Design a data structure that supports the following two operations:

void addWord(word)
bool search(word)
search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.

Example:

addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true

Note:
You may assume that all words are consist of lowercase letters a-z.
"""


class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        curr = self.root
        for char in word:
            if char not in curr:
                curr[char] = {}
            curr = curr[char]
        curr['*'] = '*'

        # time O(n)
        # space O(n)

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        curr = self.root
        for i in range(len(word)):
            char = word[i]
            if char == '.':
                for char2 in curr:
                    if char2 != '*':
                        if self.search(word[:i] + char2 + word[i + 1:]):
                            return True
                return False

            if char not in curr:
                return False
            curr = curr[char]

        return '*' in curr

        # time O(n)
        # space O(1)


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)

# -----------------------------------------------------------------------
"""
Validate that a sequence of coins does not violate the property: no more than 2 in a row face the same direction.

Examples:
TFTFTF- OK
TTFTTF - OK
FFTFFT Ok

TTT- Invalid
FFF - Invalid
FTTTF - Invalid
TFFFT - Invalid

Follow-up:
What if k faces?
"""


def validateCoins(coins, k):
    i = 0
    while i < len(coins):
        currCoin = coins[i]
        counter = 0
        while i < len(coins) and coins[i] == currCoin:
            i += 1
            counter += 1
        if counter > k:
            return False

    return True

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
Given an array of numbers, find the largest interval which is a "mountain": the two endpoints are both strictly 
smaller than all elements in between.

For example:
[2, 4, 5, 3, 6, 4, 7, 3]
The interval [0, 3] is a mountain, but [3, 7] is the largest mountain. [0, 7] is not a mountain because 3 == 3.
"""


def largestMountain(nums):
    stack1, stack2 = [], []
    res = [0, 0]

    for i in range(len(nums)):
        while stack1 and nums[i] < nums[stack1[-1]]:
            stack1.pop()
        stack1.append(i)

    for i in range(len(nums) - 1, -1, -1):
        while stack2 and nums[i] < nums[stack2[-1]]:
            stack2.pop()
        stack2.append(i)

    for i in range(1, len(stack1)):
        currLength = stack1[i] - stack1[i - 1] + 1
        if currLength > 2:
            if currLength > res[1] - res[0] + 1:
                res = [stack1[i - 1], stack1[i]]

    for i in range(len(stack2) - 1):
        currLength = stack2[i] - stack2[i + 1] + 1
        if currLength > 2:
            if currLength > res[1] - res[0] + 1:
                res = [stack2[i + 1], stack2[i]]

    return -1 if res[0] == 0 and res[1] == 0 else res

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
845. Longest Mountain in Array

Let's call any (contiguous) subarray B (of A) a mountain if the following properties hold:

B.length >= 3
There exists some 0 < i < B.length - 1 such that B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]
(Note that B could be any subarray of A, including the entire array A.)

Given an array A of integers, return the length of the longest mountain.

Return 0 if there is no mountain.

Example 1:

Input: [2,1,4,7,3,2,5]
Output: 5
Explanation: The largest mountain is [1,4,7,3,2] which has length 5.
Example 2:

Input: [2,2,2]
Output: 0
Explanation: There is no mountain.
Note:

0 <= A.length <= 10000
0 <= A[i] <= 10000
Follow up:

Can you solve it using only one pass?
Can you solve it in O(1) space?
"""


class Solution:
    def longestMountain(self, A):
        n = len(A)
        if n < 3:
            return 0

        start, end = 0, 0
        res = 0
        while start < n:
            end = start
            if end < n - 1 and A[end] < A[end + 1]:
                while end < n - 1 and A[end] < A[end + 1]:
                    end += 1
                if end < n - 1 and A[end] > A[end + 1]:
                    while end < n - 1 and A[end] > A[end + 1]:
                        end += 1
                    res = max(res, (end - start) + 1)
            start = max(end, start + 1)

        return res

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
Confusing Number II

You are running an auction with n auctioneers. Each auctioneer is assigned a card with a unique number between 1 to n. Of these some cards may be flipped upside down. Write a program that returns all of the numbers between 1 and n that may be mistaken for another possible number.

Pairs of numbers that could be confused for each other include:
"108" and "801"
"68" and "89"
"""

SWAP = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}


def confusingNum(n):
    nums = set()
    res = []
    for num in range(1, n + 1):
        rev = reverseNum(num)
        rev = swapNum(rev)
        if rev:
            if rev[0] == 0:
                nums.add(num)
                continue
            temp = 0
            for digit in rev:
                temp = temp * 10 + digit
            if temp in nums:
                res.append([temp, num])
        nums.add(num)

    return res

    # time O(n)
    # space O(n)


def reverseNum(num):
    rev = []
    while num > 0:
        rev.append(num % 10)
        num //= 10
    return rev


def swapNum(num):
    for i in range(len(num)):
        digit = num[i]
        if digit not in SWAP:
            return []
        num[i] = SWAP[digit]
    return num


# -----------------------------------------------------------------------
"""
1368. Minimum Cost to Make at Least One Valid Path in a Grid

Given a m x n grid. Each cell of the grid has a sign pointing to the next cell you should visit if you are currently in this cell. The sign of grid[i][j] can be:
1 which means go to the cell to the right. (i.e go from grid[i][j] to grid[i][j + 1])
2 which means go to the cell to the left. (i.e go from grid[i][j] to grid[i][j - 1])
3 which means go to the lower cell. (i.e go from grid[i][j] to grid[i + 1][j])
4 which means go to the upper cell. (i.e go from grid[i][j] to grid[i - 1][j])
Notice that there could be some invalid signs on the cells of the grid which points outside the grid.

You will initially start at the upper left cell (0,0). A valid path in the grid is a path which starts from the upper left cell (0,0) and ends at the bottom-right cell (m - 1, n - 1) following the signs on the grid. The valid path doesn't have to be the shortest.

You can modify the sign on a cell with cost = 1. You can modify the sign on a cell one time only.

Return the minimum cost to make the grid have at least one valid path.

Example 1:

Input: grid = [[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]
Output: 3
Explanation: You will start at point (0, 0).
The path to (3, 3) is as follows. (0, 0) --> (0, 1) --> (0, 2) --> (0, 3) change the arrow to down with cost = 1 --> (1, 3) --> (1, 2) --> (1, 1) --> (1, 0) change the arrow to down with cost = 1 --> (2, 0) --> (2, 1) --> (2, 2) --> (2, 3) change the arrow to down with cost = 1 --> (3, 3)
The total cost = 3.

"""

DIRS = [[0, 1], [0, -1], [1, 0], [-1, 0]]


class Solution:
    def minCost(self, grid):
        n = len(grid)
        m = len(grid[0])
        res = {}
        q = collections.deque()

        dfs(grid, res, 0, 0, q, 0)
        while len(q) > 0:
            curr = q.popleft()
            row, col, cost = curr[0], curr[1], curr[2] + 1
            for i in range(4):
                nextRow = row + DIRS[i][0]
                nextCol = col + DIRS[i][1]
                if isValid(n, m, nextRow, nextCol):
                    dfs(grid, res, nextRow, nextCol, q, cost)
        return res[str(n - 1) + ' ' + str(m - 1)]


def dfs(grid, res, row, col, q, cost):
    key = str(row) + ' ' + str(col)
    if key in res:
        return
    res[key] = cost
    q.append([row, col, cost])
    nextDir = DIRS[grid[row][col] - 1]
    nextRow, nextCol = row + nextDir[0], col + nextDir[1]
    if isValid(len(grid), len(grid[0]), nextRow, nextCol):
        dfs(grid, res, nextRow, nextCol, q, cost)


def isValid(n, m, row, col):
    if row < 0 or col < 0 or row >= n or col >= m:
        return False
    return True


# -----------------------------------------------------------------------
"""
Given a collection of boxes. Return the max number of boxes that you can russian doll.
Each box has (w, h, l).

Example:

Input:

[
	[3,9,9],
	[1,4,10],
	[5,10,11],
	[3,9,3],
	[1,5,3]
	[7, 12, 1]
]
Output: 3
Explanation: [1,5,3] fits in [3,9,9] which fits in [5,10,11]
All the dimensions must be strictly smaller to fit into a larger box -- [1,5,3] does not fit into [3,9,3]
"""


def numOfBoxes(dims):
    dims.sort(key=lambda x: (x[0], x[1], x[2]))
    print(dims)
    memo = [0 for i in range(len(dims))]
    res = 0
    for i in range(len(dims)):
        memo[i] = 1
        for j in range(i):
            if dims[j][0] < dims[i][0] and dims[j][1] < dims[i][1] and dims[j][2] < dims[i][2]:
                memo[i] = max(memo[i], memo[j] + 1)
        res = max(res, memo[i])

    return res

    # time O(n^2)
    # space O(n)


# -----------------------------------------------------------------------
"""

"""
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
