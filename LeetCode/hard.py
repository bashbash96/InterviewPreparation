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
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        if not root:
            return ''

        q = [root]
        res = []
        while len(q) > 0:
            curr_node = q.pop(0)
            if curr_node:
                res.append(str(curr_node.val))
                q.append(curr_node.left)
                q.append(curr_node.right)
            else:
                res.append('#')

        return ' '.join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        if len(data) == 0:
            return None
        data = data.split()
        res = TreeNode(data[0])
        curr_level, idx = [res], 1
        while len(curr_level) > 0 and idx < len(data):
            next_level = []

            for node in curr_level:
                if data[idx] != '#':
                    node.left = TreeNode(data[idx])
                    next_level.append(node.left)
                idx += 1
                if data[idx] != '#':
                    node.right = TreeNode(data[idx])
                    next_level.append(node.right)
                idx += 1
            curr_level = next_level

        return res

    # time O(n)
    # space O(n)


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
