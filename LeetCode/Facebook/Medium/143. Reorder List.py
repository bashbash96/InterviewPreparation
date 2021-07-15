"""
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.



Example 1:


Input: head = [1,2,3,4]
Output: [1,4,2,3]
Example 2:


Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]


Constraints:

The number of nodes in the list is in the range [1, 5 * 104].
1 <= Node.val <= 1000
"""


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """

        if not head or not head.next:
            return head

        fast, slow = head.next, head
        prev = slow

        while fast:
            fast = fast.next
            if fast:
                fast = fast.next

            prev = slow
            slow = slow.next

        prev.next = None

        l1 = head
        l2 = reverse_list(slow)

        res = ListNode('DUMMY')
        curr = res

        flag = True

        while l1 and l2:
            if flag:
                curr.next = l1
                l1 = l1.next
                flag = False
            else:
                curr.next = l2
                l2 = l2.next
                flag = True

            curr = curr.next

        if l1:
            curr.next = l1

        if l2:
            curr.next = l2

        return res.next

    # time O(n)
    # space O(1)


def reverse_list(head):
    prev = None
    curr = head

    while curr:
        next_ = curr.next
        curr.next = prev
        prev = curr
        curr = next_

    return prev
