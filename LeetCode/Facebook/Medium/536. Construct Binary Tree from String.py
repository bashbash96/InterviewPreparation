"""
You need to construct a binary tree from a string consisting of parenthesis and integers.

The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.

You always start to construct the left child node of the parent first if it exists.



Example 1:


Input: s = "4(2(3)(1))(6(5))"
Output: [4,2,6,3,1,5]
Example 2:

Input: s = "4(2(3)(1))(6(5)(7))"
Output: [4,2,6,3,1,5,7]
Example 3:

Input: s = "-4(2(3)(1))(6(5)(7))"
Output: [-4,2,6,3,1,5,7]


Constraints:

0 <= s.length <= 3 * 104
s consists of digits, '(', ')', and '-' only.
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def str2tree(self, s):
        """
        :type s: str
        :rtype: TreeNode
        """

        if not s:
            return None

        return construct_tree(s, 0)[0]

    # time O(n)
    # space O(h)


def construct_tree(s, idx):
    if idx >= len(s):
        return None, idx

    val, idx = get_num(s, idx)

    curr = TreeNode(val)

    if idx < len(s) and s[idx] == '(':
        curr.left, idx = construct_tree(s, idx + 1)

    if curr.left and idx < len(s) and s[idx] == '(':
        curr.right, idx = construct_tree(s, idx + 1)

    return curr, idx + 1 if idx < len(s) and s[idx] == ')' else idx


def get_num(s, idx):
    sign = 1

    if s[idx] == '-':
        sign = -1
        idx = idx + 1

    number = 0
    while idx < len(s) and s[idx].isdigit():
        number = number * 10 + int(s[idx])
        idx += 1

    return number * sign, idx
