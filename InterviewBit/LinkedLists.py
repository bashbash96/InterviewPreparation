# -----------------------------------------------------------------------
"""
Sort List

Sort a linked list in O(n log n) time using constant space complexity.

Example :

Input : 1 -> 5 -> 4 -> 3

Returned list : 1 -> 3 -> 4 -> 5

"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # @param A : head node of linked list
    # @return the head node in the linked list
    def sortList(self, A):
        return merge_sort(A)

    # time O(n*log(n))
    # space O(n)


def merge_sort(head):
    if not head or not head.next:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    if not slow:
        return head

    head1 = head
    head2 = slow.next
    slow.next = None
    head1 = merge_sort(head1)
    head2 = merge_sort(head2)

    return merge(head1, head2)


def merge(head1, head2):
    p1, p2 = head1, head2
    res = ListNode('DUMMY')
    curr = res
    while p1 or p2:
        if not p2:
            curr.next = p1
            break
        if not p1:
            curr.next = p2
            break

        if p1.val < p2.val:
            curr.next = p1
            p1 = p1.next
        else:
            curr.next = p2
            p2 = p2.next
        curr = curr.next

    return res.next


# -----------------------------------------------------------------------
"""
Insertion Sort List

Sort a linked list using insertion sort.

We have explained Insertion Sort at Slide 7 of Arrays

Insertion Sort Wiki has some details on Insertion Sort as well.

Example :

Input : 1 -> 3 -> 2

Return 1 -> 2 -> 3
"""


class Solution:
    # @param A : head node of linked list
    # @return the head node in the linked list
    def insertionSortList(self, A):
        if not A or not A.next:
            return A

        curr = A.next
        while curr:
            A = insert(A, curr)
            curr = curr.next

        return A

    # time O(n^2)
    # space O(1)


def insert(head, curr):
    left = head
    while left != curr:
        if left.val > curr.val:
            left.val, curr.val = curr.val, left.val

        left = left.next

    return head

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
