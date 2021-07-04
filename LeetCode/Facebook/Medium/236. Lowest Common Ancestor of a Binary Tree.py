"""
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”



Example 1:


Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
Example 2:


Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
Example 3:

Input: root = [1,2], p = 1, q = 2
Output: 1


Constraints:

The number of nodes in the tree is in the range [2, 105].
-109 <= Node.val <= 109
All Node.val are unique.
p != q
p and q will exist in the tree.
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        self.res = None

        def search(node, p, q):
            if not node:
                return False, False

            left = search(node.left, p, q)
            right = search(node.right, p, q)

            if self.res:
                return True, True

            if left[0] and right[1] or left[1] and right[0]:
                self.res = node
                return True, True

            if node.val == p.val:
                if right[1] or left[1]:
                    self.res = node
                    return True, True
                return True, False

            if node.val == q.val:
                if left[0] or right[0]:
                    self.res = node
                    return True, True

                return False, True

            return left[0] or right[0], left[1] or right[1]

        search(root, p, q)

        return self.res

    # time O(n)
    # space O(h)
