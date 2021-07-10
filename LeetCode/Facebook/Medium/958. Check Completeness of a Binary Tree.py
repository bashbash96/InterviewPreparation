"""
Given the root of a binary tree, determine if it is a complete binary tree.

In a complete binary tree, every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.



Example 1:


Input: root = [1,2,3,4,5,6]
Output: true
Explanation: Every level before the last is full (ie. levels with node-values {1} and {2, 3}), and all nodes in the last level ({4, 5, 6}) are as far left as possible.
Example 2:


Input: root = [1,2,3,4,5,null,7]
Output: false
Explanation: The node with value 7 isn't as far left as possible.


Constraints:

The number of nodes in the tree is in the range [1, 100].
1 <= Node.val <= 1000
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isCompleteTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        if not root:
            return True

        height = 0
        found_not_complete = False

        queue = deque()
        queue.append(root)

        while queue:

            level_length = len(queue)

            while level_length > 0:
                curr = queue.popleft()
                if not curr:
                    found_not_complete = True
                    level_length -= 1
                    continue
                else:
                    if found_not_complete:
                        return False

                queue.append(curr.left)
                queue.append(curr.right)

                level_length -= 1

            height += 1

        return True

    # time O(n)
    # space O(w)
