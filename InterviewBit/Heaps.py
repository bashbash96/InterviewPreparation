# -----------------------------------------------------------------------
"""
Merge K Sorted Lists

Merge k sorted linked lists and return it as one sorted list.

Example :

1 -> 10 -> 20
4 -> 11 -> 13
3 -> 8 -> 9
will result in

1 -> 3 -> 4 -> 8 -> 9 -> 10 -> 11 -> 13 -> 20

"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

import heapq


class Solution:
    # @param A : list of linked list
    # @return the head node in the linked list
    def mergeKLists(self, A):
        min_heap = []
        for list_ in A:
            heapq.heappush(min_heap, (list_.val, list_))
        # print(min_heap)
        res = ListNode('DUMMY')
        curr = res
        while len(min_heap) > 0:
            min_node = heapq.heappop(min_heap)
            curr.next = min_node[1]
            curr = curr.next
            next_node = min_node[1].next
            if next_node:
                heapq.heappush(min_heap, (next_node.val, next_node))

        return res.next

    # time O(n * m * log(n)), n : num of lists, m : the longest list
    # space O(1)

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
