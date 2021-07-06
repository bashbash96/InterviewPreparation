"""
Given the root of a binary tree, return the vertical order traversal of its nodes' values. (i.e., from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.



Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Example 2:


Input: root = [3,9,8,4,0,1,7]
Output: [[4],[9],[3,0,1],[8],[7]]
Example 3:


Input: root = [3,9,8,4,0,1,7,null,null,null,2,5]
Output: [[4],[9,5],[3,0,1],[8,2],[7]]
Example 4:

Input: root = []
Output: []


Constraints:

The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import defaultdict


class Solution(object):
    def verticalOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if not root:
            return []

        self.order = defaultdict(list)
        self.min_order, self.max_order = 0, 0

        def vertical_order(node, order, level):
            if not node:
                return

            self.min_order = min(self.min_order, order)
            self.max_order = max(self.max_order, order)

            self.order[order].append((level, node.val))
            vertical_order(node.left, order - 1, level + 1)
            vertical_order(node.right, order + 1, level + 1)

        vertical_order(root, 0, 0)

        res = []
        for key in range(self.min_order, self.max_order + 1):
            curr_list = [val for level, val in sorted(self.order[key], key=lambda x: x[0])]
            res.append(curr_list[:])

        return res

    # time O(n + w * hlog(h)) w: width of the tree, h: height of the tree
    # space O(n)


"""

hirizontal order:

0 -> [3]
1 -> [9, 8]
2 -> [4, 0, 1, 7]

vertical order:

0 -> [3, 0, 1]
-1 -> [9]
-2 -> [4]
1 -> [8]
2 -> [7]


[4], [9], [3, 0, 1], [8], [7]




----------------------

-2 -> [4]
-1 -> [9, 5]
0 -> [3, 0, 1]
1 -> [8, 2]
2 -> [7]

"""
