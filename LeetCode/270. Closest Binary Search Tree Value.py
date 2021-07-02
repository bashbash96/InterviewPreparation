"""
Given the root of a binary search tree and a target value, return the value in the BST that is closest to the target.



Example 1:


Input: root = [4,2,5,1,3], target = 3.714286
Output: 4
Example 2:

Input: root = [1], target = 4.428571
Output: 1
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def closestValue(self, root, target):
        """
        :type root: TreeNode
        :type target: float
        :rtype: int
        """

        res = None

        curr = root

        while curr:
            if res is None or abs(curr.val - target) < abs(res - target):
                res = curr.val

            if target <= curr.val:
                curr = curr.left
            else:
                curr = curr.right

        return res

    # time O(h)
    # space O(1)
