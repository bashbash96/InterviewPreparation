# -----------------------------------------------------------------------
"""
Convert Sorted List to Binary Search Tree
Asked in:
Google
Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

 A height balanced BST : a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
Example :


Given A : 1 -> 2 -> 3
A height balanced BST  :

      2
    /   \
   1     3

"""


# Definition for a  binary tree node
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # @param A : head node of linked list
    # @return the root node in the tree
    def sortedListToBST(self, A):
        if not A:
            return A

        left_list, curr_node, right_list = split_list(A)
        root = TreeNode(curr_node.val)
        root.left = self.sortedListToBST(left_list)
        root.right = self.sortedListToBST(right_list)

        return root

    # time O(n)
    # space O(n)


def split_list(head):
    if not head:
        return None, None, None
    if not head.next:
        return None, head, None

    slow, fast = head, head.next
    prev = slow
    while fast:
        fast = fast.next
        if fast:
            fast = fast.next
        prev = slow
        slow = slow.next

    prev.next = None
    next_ = slow.next

    return head, slow, next_

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
