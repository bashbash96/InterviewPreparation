# -----------------------------------------------------------------------
"""
122. Best Time to Buy and Sell Stock II
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:

Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Example 2:

Input: [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
             Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
             engaging multiple transactions at the same time. You must sell before buying again.
Example 3:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
"""


class Solution:
    def maxProfit(self, prices):
        if len(prices) < 1:
            return 0
        low, high = prices[0], prices[0]
        profit = 0
        i = 0
        while i < len(prices) - 1:
            while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
                i += 1
            low = prices[i]

            while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
                i += 1
            high = prices[i]

            profit += high - low

        return profit

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
350. Intersection of Two Arrays II

Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Note:

Each element in the result should appear as many times as it shows in both arrays.
The result can be in any order.

Follow up:
What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to nums2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?
"""


class Solution:

    # def intersect(self, nums1, nums2):
    #
    #     if len(nums1) == 0 or len(nums2) == 0:
    #         return []
    #
    #     nums1.sort()
    #     nums2.sort()
    #     res = []
    #
    #     p1, p2 = 0, 0
    #     while p1 < len(nums1) and p2 < len(nums2):
    #         if nums1[p1] < nums2[p2]:
    #             p1 += 1
    #         elif nums1[p1] > nums2[p2]:
    #             p2 += 1
    #         else:
    #             res.append(nums1[p1])
    #             p1 += 1
    #             p2 += 1
    #
    #     return res
    #
    #     # time O(n * log(n) + m * log(m))
    #     # space O(min(n,m))
    #

    def intersect(self, nums1, nums2):
        if len(nums1) == 0 or len(nums2) == 0:
            return []

        shorter = nums1 if len(nums1) < len(nums2) else nums2
        longer = nums2 if len(nums2) > len(nums1) else nums1

        numsCount = {}
        for num in shorter:
            if num in numsCount:
                numsCount[num] += 1
            else:
                numsCount[num] = 1

        res = []
        for num in longer:
            if num in numsCount and numsCount[num] > 0:
                res.append(num)
                numsCount[num] -= 1

        return res

        # time O(n + m)
        # space O(min(n,m))


# -----------------------------------------------------------------------
"""
118. Pascal's Triangle

Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.


In Pascal's triangle, each number is the sum of the two numbers directly above it.

Example:

Input: 5
Output:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
"""


class Solution:
    def generate(self, numRows):

        if numRows <= 0:
            return []
        elif numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1, 1]]

        res = [[1], [1, 1]]

        for i in range(2, numRows):
            currRes = [1]
            j = 1
            prevIdx = i - 1
            prevList = res[prevIdx]
            while j < len(prevList):
                currNum = prevList[j] + prevList[j - 1]
                currRes.append(currNum)
                j += 1
            currRes.append(1)
            res.append(currRes)

        return res

        # time O(n^2)
        # space O(n^2)


# -----------------------------------------------------------------------
"""
191. Number of 1 Bits
Write a function that takes an unsigned integer and return the number of '1' bits it has (also known as the Hamming weight).

Example 1:

Input: 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.

Example 2:

Input: 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.

"""


class Solution:
    def hammingWeight(self, n: int) -> int:

        count = 0

        while n > 0:
            if n & 1 != 0:
                count += 1
            n >>= 1

        return count

        # time O(log(n))
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
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # def isSymmetric(self, root):
    #
    #     if not root:
    #         return True
    #
    #     q = [root]
    #
    #     while len(q) > 0:
    #
    #         if not self.isSym(q):
    #             return False
    #
    #         levelLen = len(q)
    #
    #         while levelLen > 0:
    #             curr = q.pop(0)
    #             if curr:
    #                 q.append(curr.left)
    #                 q.append(curr.right)
    #
    #             levelLen -= 1
    #
    #     return True
    #
    # def isSym(self, level):
    #     start, end = 0, len(level) - 1
    #
    #     while start < end:
    #         left = level[start]
    #         right = level[end]
    #
    #         if left and right:
    #             if left.val != right.val:
    #                 return False
    #         elif (left and not right) or (not left and right):
    #             return False
    #
    #         start += 1
    #         end -= 1
    #
    #     return True
    #
    #     # time O(n)
    #     # space (O(maximum level length))
    #
    def isSymmetric(self, root):

        return self.recurIsSym(root, root)

        # time O(n)
        # space O(n)

    def recurIsSym(self, root1, root2):
        if not root1 and not root2:
            return True

        if not root1 or not root2:
            return False

        return root1.val == root2.val and (
                    self.recurIsSym(root1.left, root2.right) and self.recurIsSym(root1.right, root2.left))

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
