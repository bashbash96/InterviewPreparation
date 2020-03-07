# ------------------- Stack implementation with Linked List ------------------
class Node:
    def __init__(self, data):
        """
        constructor of a node
        :param data: data of the node
        """
        self.data = data
        self.next = None


class Stack:
    def __init__(self):
        """
        constructor of empty stack
        """
        self.head = None
        self.size = 0

    def push(self, data):
        """
        add new node to head of the stack
        :param data:
        :return:
        """
        node = Node(data)
        node.next = self.head
        self.head = node
        self.size += 1

    def pop(self):
        """
        remove node from the head of the stack
        :return: data of the removed node
        """
        data = None

        if self.head:
            data = self.head.data
            self.head = self.head.next
            self.size -= 1

        return data

    def length(self):
        """
        :return: the length of the stack
        """
        return self.size

    def isEmpty(self):
        """
        check if the stack empty or not
        :return: True / False
        """
        return self.size == 0

    def peek(self):
        """
        :return: the data of the head node
        """
        if self.head:
            return self.head.data

    def deleteStack(self):
        """
        delete the entire stack
        :return:
        """
        self.head = None

    def printStack(self):
        """
        print the stack elements
        :return:
        """
        if not self.head:
            return

        curr = self.head
        while curr.next:
            print(curr.data, end=" -> ")
            curr = curr.next

        if curr:
            print(curr.data)


# ------------------- Queue implementation with Linked List ------------------
class Queue:
    def __init__(self):
        """
        constructor of empty Queue
        """
        self.head = None
        self.tail = None
        self.length = 0

    def enqueue(self, data):
        """
        add new node to the queue
        :param data: data to insert
        :return:
        """
        node = Node(data)
        self.length += 1
        if not self.head:
            self.head = self.tail = node
            return

        self.tail.next = node
        self.tail = node

    def dequeue(self):
        """
        remove the first element of the queue
        :return: data / None
        """
        if self.head is None:
            return None

        data = self.head.data
        self.head = self.head.next
        self.length -= 1
        return data

    def isEmpty(self):
        """
        check if the queue is empty or note
        :return: True / False
        """
        return self.length == 0

    def deleteQueue(self):
        """
        delete the entire queue
        :return:
        """
        self.head = self.tail = None

    def peek(self):
        """
        :return: the data of the head node
        """
        if self.head:
            return self.head.data

        return None

    def printQueue(self):
        """
        print the queue elements
        :return:
        """
        curr = self.head

        while curr.next is not None:
            print(curr.data, end=" -> ")
            curr = curr.next

        if curr is not None:
            print(curr.data)


# -----------------------------------------------------------------------
"""
3.2 Stack Min: How would you design a stack which, in addition to push and pop, has a function min
which returns the minimum element? Push, pop and min should all operate in 0(1) time.
"""


class NodeMin:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.min = data


class MinStack:
    def __init__(self):
        self.head = None

    def push(self, data):
        newNode = NodeMin(data)

        if not self.head:
            self.head = newNode
        else:
            newNode.next = self.head
            if data > self.head.min:
                newNode.min = self.head.min
            self.head = newNode

    def getMin(self):
        """
        get the min of the stack
        :return:
        """
        if self.head:
            return self.head.min
        return None

        # time O(1)
        # space O(1)

    def printStack(self):
        """
        print the stack elements
        :return:
        """
        if not self.head:
            return

        curr = self.head
        while curr.next:
            print('data', curr.data, 'min', curr.min, end=" -> ")
            curr = curr.next

        if curr:
            print(curr.data)


# -----------------------------------------------------------------------
"""
3.3 Stack of Plates: Imagine a (literal) stack of plates. If the stack gets too high, it might topple.
Therefore, in real life, we would likely start a new stack when the previous stack exceeds some
threshold. Implement a data structure SetOfStacks that mimics this. SetOfStacks should be
composed of several stacks and should create a new stack once the previous one exceeds capacity.
SetOfStacks. push () and SetOfStacks. pop() should behave identically to a single stack
(that is, pop ( ) should return the same values as it would if there were just a single stack).
FOLLOW UP
Implement a function popAt (int index) which performs a pop operation on a specific substack.
"""


class SetOfStacks:
    def __init__(self, threshold):
        self.set = []
        self.threshold = threshold

    def push(self, data):

        if len(self.set) == 0:
            self.set.append(Stack())

        lastStack = self.set[len(self.set) - 1]

        if lastStack.length() < self.threshold:
            lastStack.push(data)
        else:
            lastStack = Stack()
            lastStack.push(data)
            self.set.append(lastStack)

        # time O(1)
        # space O(1)

    def pop(self):

        if len(self.set) == 0:
            return None

        lastStack = self.set[len(self.set) - 1]

        data = lastStack.pop()
        if not lastStack.head:
            self.set.pop()

        return data

        # time O(1)
        # space O(1)

    def printStacks(self):

        for stack in self.set:
            stack.printStack()

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
3.4 Queue via Stacks: Implement a MyQueue class which implements a queue using two stacks.
"""


class MyQueue:
    def __init__(self):
        """
        constructor for MyQueue class which implemented with two stacks
        """
        self.s1 = Stack()
        self.s2 = Stack()

    def enqueue(self, data):
        """
        add element to the queue
        :param data:
        :return:
        """
        self.s1.push(data)

        # time O(1)

    def dequeue(self):
        """
        remove the first  element from the queue
        :return:
        """
        data = None

        if self.s2.length() > 0:  # if there is elements in the second stack pop the head
            data = self.s2.pop()
        elif self.s1.length() > 0:  # else move all elements from the first stack to the second then pop the head
            self.moveElements()
            data = self.s2.pop()

        return data

        # time  -> the worst case is O(n) but it's rare so the average is O(1)
        # space O(1)

    def peek(self):
        """
        return the value of the first element of the queue
        :return:
        """
        if self.s2.length() > 0:
            return self.s2.peek()
        elif self.s1.length() > 0:
            self.moveElements()
            return self.s2.peek()
        else:
            return None

    def length(self):
        """
        how many elements in the queue
        :return:
        """
        return self.s1.length() + self.s2.length()

    def moveElements(self):
        """
        move all the elements from the first stack into the second stack
        :return:
        """
        while self.s1.head:
            self.s2.push(self.s1.pop())

    def printQueue(self):
        """
        print all the queue elements
        :return:
        """
        if self.s1.length() < 1 and self.s2.length() < 1:
            return

        curr = self.s2.head
        if curr:
            while curr.next:
                print(curr.data, end=" -> ")
                curr = curr.next

            if self.s1.length() == 0:
                print(curr.data)
                return
            else:
                print(curr.data, end=" -> ")

        s1 = self.s1.head
        res = []
        while s1:
            res.append(s1.data)
            s1 = s1.next

        for i in range(len(res) - 1, 0, -1):
            print(res[i], end=" -> ")

        print(res[0])


# -----------------------------------------------------------------------
"""
3.5 Sort Stack: Write a program to sort a stack such that the smallest items are on the top. You can use
an additional temporary stack, but you may not copy the elements into any other data structure
(such as an array). The stack supports the following operations: push, pop, peek, and isEmpty.
"""


def sortStack(s):
    if not s.head:
        return

    lengthToSort = getLengthToSort(s)

    while lengthToSort > 0:
        oneSortStep(s, lengthToSort)
        lengthToSort -= 1

    # time O(n^2)
    # space O(n)


def oneSortStep(s, length):
    helperStack = Stack()
    currMax = s.pop()
    while length > 0:
        curr = s.pop()
        if curr > currMax:
            currMax, curr = curr, currMax

        helperStack.push(curr)
        length -= 1

    s.push(currMax)

    while helperStack.head:
        s.push(helperStack.pop())


def getLengthToSort(s):
    helperStack = Stack()
    length = 0

    currMax = s.pop()
    while s.head:
        curr = s.pop()
        if curr > currMax:
            currMax, curr = curr, currMax
        helperStack.push(curr)
        length += 1

    s.push(currMax)
    while helperStack.head:
        s.push(helperStack.pop())

    return length - 1


# -----------------------------------------------------------------------
"""
3.6 Animal Shelter: An animal shelter, which holds only dogs and cats, operates on a strictly "first in, first
out" basis. People must adopt either the "oldest" (based on arrival time) of all animals at the shelter,
or they can select whether they would prefer a dog or a cat (and will receive the oldest animal of
that type). They cannot select which specific animal they would like. Create the data structures to
maintain this system and implement operations such as enqueue, dequeueAny, dequeueDog,
and dequeueCat. You may use the built-in LinkedList data structure.
"""


class Animal:
    def __init__(self, type=None):
        self.type = type
        self.enterTime = 0

    def setEnterTime(self, time):
        self.enterTime = time

    def __str__(self):
        return self.type


class AnimalShelter:
    def __init__(self):
        self.dogs = Queue()
        self.cats = Queue()
        self.currTime = 0

    def enqueue(self, type):
        """
        enqueue animal to the shelter

        :param type: we assume that the types are 'cat' or 'dog'
        :return:
        """
        currAnimal = Animal(type)
        currAnimal.setEnterTime(self.currTime)
        self.currTime += 1
        if type == 'dog':
            self.dogs.enqueue(currAnimal)
        else:
            self.cats.enqueue(currAnimal)

    def size(self):
        return self.dogs.length + self.cats.length

    def dequeueAny(self):
        """
        return the oldest animal, if there is not cats return the oldest dog
        and vise versa
        :return:
        """
        if self.size() == 0:
            return None
        if not self.dogs.head:
            return self.cats.dequeue()
        elif not self.cats.head:
            return self.dogs.dequeue()
        else:
            if self.dogs.peek().enterTime < self.cats.peek().enterTime:
                return self.dogs.dequeue()
            return self.cats.dequeue()

    def dequeueDog(self):

        if self.dogs.length > 0:
            return self.dogs.dequeue()
        return None

    def dequeueCat(self):

        if self.cats.length > 0:
            return self.cats.dequeue()
        return None
