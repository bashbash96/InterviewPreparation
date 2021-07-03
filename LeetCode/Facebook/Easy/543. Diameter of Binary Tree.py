"""
Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.



Example 1:


Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3is the length of the path [4,2,1,3] or [5,2,1,3].
Example 2:

Input: root = [1,2]
Output: 1


Constraints:

The number of nodes in the tree is in the range [1, 104].
-100 <= Node.val <= 100
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        return diameter_of_bin_tree(root)[1]

        # time O(n)
        # space O(h)


def diameter_of_bin_tree(node):
    if not node:
        return 0, 0

    left = diameter_of_bin_tree(node.left)
    right = diameter_of_bin_tree(node.right)

    curr_path_length = left[0] + right[0]

    longest_path = max(left[1], right[1], curr_path_length)

    return max(left[0], right[0]) + 1, longest_path
