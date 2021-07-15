"""
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.



Example 1:


Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]
Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]


Constraints:

The number of nodes in each linked list is in the range [1, 100].
0 <= Node.val <= 9
It is guaranteed that the list represents a number that does not have leading zeros.
"""


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        p1, p2 = l1, l2
        res = curr = ListNode('DUMMY')

        curr_sum = 0
        carry = 0

        while p1 or p2:
            val1 = p1.val if p1 else 0
            val2 = p2.val if p2 else 0

            curr_sum = val1 + val2 + carry
            new_node = ListNode(curr_sum % 10)
            carry = curr_sum // 10
            curr.next = new_node

            if p1:
                p1 = p1.next
            if p2:
                p2 = p2.next
            curr = curr.next

        if carry:
            curr.next = ListNode(carry)

        return res.next

    # time O(max(n, m))
    # space O(max(n, m))
