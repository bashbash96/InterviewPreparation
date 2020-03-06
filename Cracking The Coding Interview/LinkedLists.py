class Node:
    def __init__(self, data):
        """
        constructor of node
        :param data: data of the node
        """
        self.data = data
        self.next = None


class SinglyLinkedList:
    def __init__(self):
        """
        constructor of singly linked list
        """
        self.head = None
        self.tail = None
        self.length = 0

    def prepend(self, data):
        """
        add to the beginning of the list
        :param data: data to add
        :return:
        """
        newNode = Node(data)
        if self.head is None:
            self.head = self.tail = newNode
        else:
            newNode.next = self.head
            self.head = newNode

        self.length += 1

    def append(self, data):
        """
        add to end of the list
        :param data: data to add
        :return:
        """
        newNode = Node(data)

        if self.head is None:
            self.head = self.tail = newNode
        else:
            self.tail.next = newNode
            self.tail = newNode

        self.length += 1

    def size(self):
        """

        :return: the length of the linked list
        """
        return self.length

    def search(self, data):
        """
        check if certain data is in the linked list
        :param data: data to check
        :return: node / None
        """
        curr = self.head

        while curr is not None:
            if curr.data == data:
                return curr
            curr = curr.next

        return None

    def delete(self, data):
        """
        remove certain data from the linked list, it will remove
        the first appearance of the data, if there is no such a data
        the function won't do anything
        :param data: data to remove
        :return:
        """
        if self.head is None:
            return

        if self.head.data == data:
            if self.head == self.tail:
                self.head = self.tail = None
            else:
                self.head = self.head.next
            self.length -= 1
            return

        curr = self.head
        while curr.next is not None:
            if curr.next.data == data:
                if curr.next == self.tail:
                    self.tail = curr
                curr.next = curr.next.next
                self.length -= 1
                return
            curr = curr.next

    def deleteLinkedList(self):
        """
        delete the whole linked list
        :return:
        """
        self.head = self.tail = None

    def printLinkedList(self):
        """
        print the linked list elements
        :return:
        """

        if self.head is None:
            return

        curr = self.head

        while curr.next is not None:
            print(curr.data, "-> ", end='')
            curr = curr.next

        print(curr.data)

    def addList(self, lst):
        """
        add list of data to the linked list
        :param lst: list of data
        :return:
        """
        for val in lst:
            self.append(val)

    # -----------------------------------------------------------------------
    """
    2.1 Remove Dups: Write code to remove duplicates from an unsorted linked list.
    FOLLOW UP
    How would you solve this problem if a temporary buffer is not allowed?
    """

    def removeDups(self):
        curr = self.head
        prev = None
        setOfData = set()

        while curr is not None:
            if curr.data in setOfData:
                prev.next = curr.next
            else:
                setOfData.add(curr.data)
                prev = curr

            curr = curr.next

        # time O(n)
        # space O(n)

    def removeDups2(self):
        p1 = self.head

        while p1 is not None:
            p2 = p1
            while p2.next is not None:
                if p2.next.data == p1.data:
                    p2.next = p2.next.next
                else:
                    p2 = p2.next

            p1 = p1.next

        # time O(n^2)
        # space O(1)

    # -----------------------------------------------------------------------
    """
    2.2 Return Kth to Last: Implement an algorithm to find the kth to last element of a singly linked list.
    """

    def getKthToLast(self, k):

        size = self.size()
        if k > size or k < 0:
            return None

        k = size - k
        res = self.head

        i = 0
        while i < k and res is not None:
            res = res.next
            i += 1

        lst = SinglyLinkedList()
        lst.head = res

        return lst

        # time O(n)
        # space O(1)

    def getKthToLast2(self, k):
        p1 = p2 = self.head

        if k < 0:
            return None

        i = 0
        while i < k and p2 is not None:
            p2 = p2.next
            i += 1

        while p2 is not None:
            p1 = p1.next
            p2 = p2.next

        lst = SinglyLinkedList()
        lst.head = p1

        return lst

        # time O(n)
        # space O(1)

    # -----------------------------------------------------------------------
    """
    2.3 Delete Middle Node: Implement an algorithm to delete a node in the middle (Le., any node but
    the fi rst and last node, not necessarily the exact middle) of a singly linked list, given only access to
    that node.
    EXAMPLE
    Input: the node c from the linked list a - >b- >c - >d - >e- >f
    Result: nothing is returned, but the new linked list looks like a->b->d->e->f
    """

    def deleteMid(self, node):

        if node is None or node.next is None:
            return False

        node.data = node.next.data
        node.next = node.next.next

        return True

        # time O(1)
        # space O(1)

    # -----------------------------------------------------------------------
    """
    2.4 Partition: Write code to partition a linked list around a value x, such that all nodes less than x come
    before all nodes greater than or equal to x. If x is contained within the list, the values of x only need
    to be after the elements less than x (see below). The partition element x can appear anywhere in the
    "right partition"; it does not need to appear between the left and right partitions.
    EXAMPLE
    Input: 3 -) 5 -) 8 -) 5 -) 113 -) 2 -) 1 [partition = 5]
    Output: 3 -) 1 -) 2 -) 113 -) 5 -) 5 -) 8
    """

    def partition(self, x):

        if self.head is None:
            return None

        res = SinglyLinkedList()

        curr = self.head

        while curr is not None:
            if curr.data < x:
                res.prepend(curr.data)
            else:
                res.append(curr.data)

            curr = curr.next

        return res

        # time O(n)
        # space O(n)

    # -----------------------------------------------------------------------
    """
    Sum Lists: You have two numbers represented by a linked list, where each node contains a single
    digit. The digits are stored in reverse order, such that the 1 's digit is at the head of the list. Write a
    function that adds the two numbers and returns the sum as a linked list.
    EXAMPLE
    Input: (7-> 1 -> 6) + (5 -> 9 -> 2) .Thatis,617 + 295.
    Output: 2 - > 1 - > 9. That is, 912.
    FOLLOW UP
    Suppose the digits are stored in forward order. Repeat the above problem.
    Input: (6 -> 1 -> 7) + (2 -> 9 -> 5).Thatis,617 + 295.
    Output: 9 - > 1 - > 2. That is, 912.
    """

    def sumLists(self, other):

        if self.head is None:
            return other
        elif other.head is None:
            return self

        p1, p2 = self.head, other.head

        sum, carry = 0, 0
        res = SinglyLinkedList()

        while p1 and p2:
            sum = p1.data + p2.data + carry

            carry = sum // 10
            sum = sum % 10
            res.append(sum)
            p1 = p1.next
            p2 = p2.next

        while p1:
            sum = p1.data + carry
            carry = sum // 10
            sum = sum % 10
            res.append(sum)

        while p2:
            sum = p2.data + carry
            carry = sum // 10
            sum = sum % 10
            res.append(sum)

        if carry != 0:
            res.append(carry)

        return res

        # time O(n + m)
        # space O(max(n,m))

    # -----------------------------------------------------------------------
    """
    2.6 Palindrome: Implement a function to check if a linked list is a palindrome.
    """

    def isPalindrome(self):

        if self.head is None:
            return True

        reversedList = self.reverse(self)

        return self.isEqual(self, reversedList)

        # time O(n)
        # space O(n)

    def isEqual(self, lst1, lst2):

        p1, p2, = lst1.head, lst2.head

        while p1 and p2:
            if p1.data != p2.data:
                return False

            p1 = p1.next
            p2 = p2.next

        return True

    def reverse(self, lst):
        reversedList = SinglyLinkedList()

        curr = lst.head

        while curr:
            reversedList.prepend(curr.data)
            curr = curr.next

        return reversedList

    # -----------------------------------------------------------------------
    """
    2.7 Intersection: Given two (singly) linked lists, determine if the two lists intersect. Return the
    intersecting node. Note that the intersection is defined based on reference, not value. That is, if the
    kth node of the first linked list is the exact same node (by reference) as the jth node of the second
    linked list, then they are intersecting.
    """

    def intersection(self, other):
        """
        check intersection between two linked lists with hash set data structure
        :param other:
        :return:
        """

        if not self.head or not other.head:
            return None

        nodes = self.getNodesSet(self)

        return self.checkIntersect(nodes, other)

        # time O(n + m)
        # space O(n)

    def checkIntersect(self, nodes, lst):

        curr = lst.head
        while curr:
            if curr in nodes:
                return curr
            curr = curr.next

        return None

    def getNodesSet(self, lst):
        nodes = set()
        curr = lst.head

        while curr:
            nodes.add(curr)
            curr = curr.next

        return nodes

    def intersection2(self, other):
        """
        check two linked lists intersection with no extra space
        :param other:
        :return:
        """
        if not self.head or not other.head:
            return None

        tail1, length1 = self.getTailAndLength(self)
        tail2, length2 = self.getTailAndLength(other)

        if tail1 != tail2:
            return None

        diff = 0
        if length1 > length2:
            longer = self.head
            shorter = other.head
            diff = length1 - length2
        else:
            longer = other.head
            shorter = self.head
            diff = length2 - length1

        longer = self.movePointer(longer, diff)

        while longer and shorter:
            if longer == shorter:
                return shorter
            longer = longer.next
            shorter = shorter.next

        return None

    # time O(n + m)
    # space O(1)

    def movePointer(self, lst, diff):

        curr = lst
        while diff > 0:
            curr = curr.next
            diff -= 1

        return curr

    def getTailAndLength(self, lst):
        curr = lst.head
        length = 0

        while curr.next:
            length += 1
            curr = curr.next

        return curr, length + 1

    # -----------------------------------------------------------------------
    """
    2.8 Loop Detection: Given a circular linked list, implement an algorithm that returns the node at the
    beginning of the loop.
    DEFINITION
    Circular linked list: A (corrupt) linked list in which a node's next pointer points to an earlier node, so
    as to make a loop in the linked list.
    EXAMPLE
    Input: A -> B -> C - > D -> E -> C [the same C as earlier)
    Output: C
    """

    def loopDetection(self):

        if not self.head:
            return None

        fast = slow = self.head

        while fast and slow:
            fast = fast.next
            if fast:
                fast = fast.next
            slow = slow.next
            if slow == fast:
                break

        slow = self.head

        while fast and slow:
            if slow == fast:
                return fast
            slow = slow.next
            fast = fast.next

        return None

        # time O(n)
        # space O(1)

    # -----------------------------------------------------------------------
