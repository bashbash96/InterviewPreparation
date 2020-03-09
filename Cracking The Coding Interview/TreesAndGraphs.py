import sys
import random


# ------------------- Linked List Trees ------------------
class Node:
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None


class Node2:
    def __init__(self, data):
        self.data = data
        self.next = None


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
        node = Node2(data)
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


# ------------------- Binary Search Tree implementation with Linked List ------------------
class BST:
    def __init__(self):
        self.root = None

    def insert(self, data):
        if self.root is None:
            self.root = Node(data)
        else:
            self.insert2(self.root, data)

    def insert2(self, node, data):

        if data > node.data:
            if node.right is None:
                node.right = Node(data)
            else:
                self.insert2(node.right, data)
        else:
            if node.left is None:
                node.left = Node(data)
            else:
                self.insert2(node.left, data)

    # ------------------- DFS Traversal ------------------
    def inOrder(self):
        """
        in order traversal to a binary tree
        :return:
        """
        self.inOrder2(self.root)
        print()

    def inOrder2(self, node):
        if node is None:
            return

        self.inOrder2(node.left)
        print(node.data, end=" ")
        self.inOrder2(node.right)

    def preOrder(self):
        """
        pre order traversal to a binary tree
        :return:
        """
        self.preOrder2(self.root)
        print()

    def preOrder2(self, node):
        if node is None:
            return

        print(node.data, end=" ")
        self.preOrder2(node.left)
        self.preOrder2(node.right)

    def postOrder(self):
        """
         post order traversal to a binary tree
        :return:
        """
        self.postOrder2(self.root)
        print()

    def postOrder2(self, node):
        if node is None:
            return

        self.postOrder2(node.left)
        self.postOrder2(node.right)
        print(node.data, end=" ")

    # ------------------- BFS Traversal ------------------
    def BFS(self):
        if self.root is None:
            return

        q = Queue()
        q.enqueue(self.root)
        while q.length > 0:
            curr = q.dequeue()
            print(curr.data, end=" ")
            if curr.left is not None:
                q.enqueue(curr.left)
            if curr.right is not None:
                q.enqueue(curr.right)

        print()

    def height(self):
        return self.height2(self.root)

    def height2(self, node):
        if node is None:
            return 0

        return 1 + max(self.height2(node.left), self.height2(node.right))

    # -----------------------------------------------------------------------
    """
    4.2 Minimal Tree: Given a sorted (increasing order) array with unique integer elements, write an
    algorithm to create a binary search tree with minimal height.
    """

    def minimalTree(self, arr):
        if len(arr) == 0:
            return None

        bst = BST()
        bst.root = self.recurMinimalTree(arr, 0, len(arr) - 1)

        return bst

    # time O(n)
    # space O(n)

    def recurMinimalTree(self, arr, start, end):

        if start > end:
            return None

        mid = (start + end) // 2
        currNode = Node(arr[mid])
        currNode.left = self.recurMinimalTree(arr, start, mid - 1)
        currNode.right = self.recurMinimalTree(arr, mid + 1, end)

        return currNode

    # -----------------------------------------------------------------------
    """
    4.3 List of Depths: Given a binary tree, design an algorithm which creates a linked list of all the nodes
    at each depth (e.g., if you have a tree with depth D, you 'll have D linked lists).
    """

    def listOfDepths(self):

        if not self.root:
            return []

        q = Queue()
        q.enqueue(self.root)
        res = []

        while not q.isEmpty():
            currList = []
            currLevelLength = q.length
            while currLevelLength > 0:
                currNode = q.dequeue()
                if currNode.left:
                    q.enqueue(currNode.left)
                if currNode.right:
                    q.enqueue(currNode.right)

                currList.append(currNode.data)

                currLevelLength -= 1

            res.append(currList)

        return res

        # time O(n)
        # space O(n)

    # -----------------------------------------------------------------------
    """
    4.4 Check Balanced: Implement a function to check if a binary tree is balanced. For the purposes of
    this question, a balanced tree is defined to be a tree such that the heights of the two subtrees of any
    node never differ by more than one.
    """

    def checkBalanced(self):

        if not self.root:
            return True

        return self.recurCheckBalanced(self.root)[1]

        # time O(n)
        # space O(h)

    def recurCheckBalanced(self, node):

        if not node:
            return (0, True)

        left = self.recurCheckBalanced(node.left)
        right = self.recurCheckBalanced(node.right)

        if not left[1] or not right[1]:
            return (max(left[0], right[0]) + 1, False)

        return (max(left[0], right[0]) + 1, abs(left[0] - right[0]) <= 1)

    # -----------------------------------------------------------------------
    """
    4.5 Validate BST: Implement a function to check if a binary tree is a binary search tree.
    """

    def checkBST(self):
        if not self.root:
            return True

        return self.recurCheckBST(self.root, -sys.maxsize, sys.maxsize)

        # time O(n)
        # space O(h)

    def recurCheckBST(self, node, minVal, maxVal):
        """
        use the definition of the bst that for each node the left sub tree
        has to be less than its value and the right sub tree has to be greater
        than its value
        :param node:
        :param minVal:
        :param maxVal:
        :return:
        """

        if not node:  # base case
            return True

        if node.data > maxVal or node.data <= minVal:
            return False

        return self.recurCheckBST(node.left, minVal, node.data) and self.recurCheckBST(node.right, node.data, maxVal)

    # -----------------------------------------------------------------------
    """
    4.6 Successor: Write an algorithm to find the "next" node (i.e., in-order successor) of a given node in a
    binary search tree. You may assume that each node has a link to its parent.
    """

    def successor(self, node):

        if node is None:
            return None

        if node.right:
            return self.getMin(node.right)

        p = node.parent  # assume that the node has access to its parent, here its not the case...

        while p:
            if p.left == node:
                return p
            node = p
            p = p.parent

        return None

        # time O(log(n))
        # space O(h)

    def getMin(self, node):
        if not node.left:
            return node.data

        return self.getMin(node.left)

    # -----------------------------------------------------------------------
    """
    4.8 First Common Ancestor: Design an algorithm and write code to find the first common ancestor
    of two nodes in a binary tree. Avoid storing additional nodes in a data structure. NOTE: This is not
    necessarily a binary search tree.
    """

    def firstCommonAncestor(self, node1, node2):
        height1, height2 = self.getHeight(node1), self.getHeight(node2)  # get height according to root
        diff = height2 - height1

        if diff < 0:
            shorter = node2
            longer = node1
        else:
            shorter = node1
            longer = node2

        longer = self.moveUp(longer, diff)

        while longer and shorter and (longer != shorter):
            longer = longer.parent
            shorter = shorter.parent

        if shorter and longer:
            return shorter

        return None

        # time O(d) -> d is the depth of the deepest node
        # space O(d)

    def getHeight(self, node):

        return self.recurGetHeight(self.root, node)

    def recurGetHeight(self, curr, node):

        if not curr:
            return -1
        if curr == node:
            return 1

        left = self.recurGetHeight(curr.left, node)
        right = self.recurGetHeight(curr.right, node)

        if left != -1 or right != -1:
            return max(left, right) + 1

        return -1

    def moveUp(self, node, diff):

        while diff > 0 and node:
            node = node.parent
            diff -= 1

        return node

    # -----------------------------------------------------------------------
    """
    4.10 Check Subtree: Tl and T2 are two very large binary trees, with Tl much bigger than T2. Create an
    algorithm to determine if T2 is a subtree of Tl.
    A tree T2 is a subtree of T1 if there exists a node n in Tl such that the subtree of n is identical to T2 .
    That is, if you cut off the tree at node n, the two trees would be identical.
    """

    def checkSubTree(self, other):

        largePre = self.getPreOrder(self.root)
        smallPre = self.getPreOrder(other.root)

        return smallPre in largePre

        # time O(n + m)
        # space O(n + m)

    def getPreOrder(self, node):

        curr = ''
        if not node:
            return 'x'

        curr += str(node.data)

        curr += self.getPreOrder(node.left)
        curr += self.getPreOrder(node.right)

        return curr

    def checkSubTree2(self, other):

        if not other.root:
            return True

        return self.recurCheckSubTree2(self.root, other.root)

        # time O(n + km) -> number of appearances of the smallest tree root in the biggest tree
        # space O(log(n) + log(m))

    def recurCheckSubTree2(self, t1, t2):

        if not t1:
            return False
        elif t1.data == t2.data and self.checkMatch(t1, t2):
            return True
        else:
            return self.recurCheckSubTree2(t1.left, t2) or self.recurCheckSubTree2(t1.right, t2)

    def checkMatch(self, t1, t2):

        if not t1 and not t2:
            return True
        elif not t1 or not t2:
            return False
        elif t1.data == t2.data:
            return self.checkMatch(t1.left, t2.left) and self.checkMatch(t1.right, t2.right)

        return False

    # -----------------------------------------------------------------------
    """
    4.11 Random Node: You are implementing a binary search tree class from scratch, which, in addition
    to insert, find, and delete, has a method getRandomNode() which returns a random node
    from the tree. All nodes should be equally likely to be chosen. Design and implement an algorithm
    for get Ra ndomNode, and explain how you would implement the rest of the methods.
    """

    def getRandomNode(self):
        if not self:
            return None

        leftSize = 0 if not self.left else self.left.size

        randomNumber = random.randrange(1, self.size + 1)

        if randomNumber == leftSize:
            return self
        elif randomNumber < leftSize:
            return self.left.getRandomNode()
        else:
            return self.right.getRandomNode()

        # time O(log(n))
        # space O(log(n))

    # -----------------------------------------------------------------------
    """
    4.12 Paths with Sum: You are given a binary tree in which each node contains an integer value (which
    might be positive or negative). Design an algorithm to count the number of paths that sum to a
    given value. The path does not need to start or end at the root or a leaf, but it must go downwards
    (traveling only from parent nodes to child nodes).
    """

    def pathsWithSum(self, targetSum):
        dp = {}

        return self.pathsWithSum2(self.root, targetSum, dp)

        # time O(nlog(n))
        # space O(log(n))

    def pathsWithSum2(self, node, targetSum, dp):

        if not node:
            return 0

        if (node, targetSum) not in dp:
            rootCount = self.recurPathsWithSum(node, targetSum, dp)
            leftCount = self.pathsWithSum2(node.left, targetSum, dp)
            rightCount = self.pathsWithSum2(node.right, targetSum, dp)

            dp[(node, targetSum)] = rootCount + leftCount + rightCount

        return dp[(node, targetSum)]

    def recurPathsWithSum(self, node, currSum, dp):

        if not node or currSum < 0:
            return 0
        currSum -= node.data
        currRes = 0

        if (node, currSum) not in dp:
            if currSum == 0:
                currRes += 1

            currRes += self.recurPathsWithSum(node.left, currSum, dp)
            currRes += self.recurPathsWithSum(node.right, currSum, dp)
            dp[(node, currSum)] = currRes

        return dp[(node, currSum)]


# ------------------- AVL Trees with Linked List ------------------

class AVL:
    def __init__(self):
        self.root = None

    def search(self, data):
        if self.root is None:
            return False

        if data < self.root.data:
            return self.search(self.root.left, data)
        elif data > self.root.data:
            return self.search(self.root.right, data)
        else:
            return True

        def rotateRight(node):
            l = node.left
            lr = l.right
            l.right = node
            node.left = lr

            return l

        def rotateLeft(node):
            r = node.rigth
            rl = r.left
            r.left = node
            node.right = rl

            return r


# ----------------- Graph implementation with list ---------------------

from sys import maxsize
from collections import defaultdict


class Graph:
    def __init__(self):
        """
        constructor for graph
        """
        self.E = defaultdict(set)
        self.v = set()

    def addEdge(self, u, v):
        """
        add new edge between two vertices (u,v), for undirected graph remove
        the commented instructions
        :param u:
        :param v:
        :return:
        """
        self.v.add(v)
        self.v.add(u)

        if u in self.E:
            self.E[u].add(v)
        else:
            self.E[u] = {v}

        # if v in self.E:
        #     self.E[v].add(u)
        # else:
        #     self.E[v] = {u}

    def BFS(self, v):
        """
        Breadth-First Search traversal on graph
        :param v:
        :return:
        """
        # initialization
        distance = {}
        color = {}
        parent = {}

        for ver in self.v:
            distance[ver] = maxsize
            color[ver] = 'w'
            parent[ver] = None

        distance[v] = 0
        color[v] = 'g'

        # creating a queue
        q = [v]

        while len(q) > 0:
            curr = q.pop(0)
            print(curr, end=' ')

            # for all adjacent nodes if there is an adjacent node that we didn't visit, visit it
            for ver in self.E[curr]:
                if color[ver] == 'w':
                    color[ver] = 'g'
                    distance[ver] = distance[curr] + 1
                    parent[ver] = curr
                    q.append(ver)

            color[curr] = 'b'

        print()

        return distance, parent

        # time O(|V| + |E|)
        # space O(|V|)

    def printPath2(self, u, v, parent):
        if u == v:
            print(v, end=' ')
        else:
            if parent[v] is None:
                print('no path')
            else:
                self.printPath2(u, parent[v], parent)
                print(v, end=' ')

    def printPath(self, u, v):
        (d, f) = self.BFS(u)

        self.printPath2(u, v, f)

        # time O(|E|)
        # space O(|E|)

    def DFS(self):
        color, parent = {}, {}

        for v in self.v:
            color[v] = 'w'
            parent[v] = None

        for v in self.v:
            if color[v] == 'w':
                self.DFSVisit(v, color, parent)

        return parent

        # time O(|V| + |E|)
        # space O(|V| + |E|)

    def DFSVisit(self, u, color, parent):
        color[u] = 'g'
        print(u, end=' ')
        for v in self.E[u]:
            if color[v] == 'w':
                parent[v] = u
                self.DFSVisit(v, color, parent)

        color[u] = 'b'

    def TopologicalSort(self):
        color = {}
        ans = []

        if self.isCyclic():
            return 'Error - can\'t do a topological sort for cyclic graph'
        for v in self.v:
            color[v] = 'w'

        for v in self.v:
            if color[v] == 'w':
                self.topologicalVisit(v, color, ans)

        ans.reverse()
        print(ans)

        # time O(|V| + |E|)
        # space O(|V| + |E|)

    def topologicalVisit(self, u, color, ans):
        color[u] = 'g'

        for v in self.E[u]:
            if color[v] == 'w':
                self.topologicalVisit(v, color, ans)

        color[u] = 'b'
        ans.append(u)

    def isCyclic(self):
        color = {}

        for v in self.v:
            color[v] = 'w'

        for v in self.v:
            if self.isCyclic2(v, color):
                return True

        return False

        # time O(|V| + |E|)
        # space O(|V| + |E|)

    def isCyclic2(self, v, color):
        color[v] = 'g'

        for u in self.E[v]:
            if color[u] == 'w':
                self.isCyclic2(u, color)
            elif color[u] == 'g':
                return True

        color[v] = 'b'
        return False

    # -----------------------------------------------------------------------
    """
    4.1 Route Between Nodes: Given a directed graph, design an algorithm to find out whether there is a
    route between two nodes.
    Hints: #127
    """

    def checkRoute(self, s, d):
        if s == d:
            return True
        color = {}

        for ver in self.v:
            color[ver] = 'w'

        color[s] = 'g'
        for adj in self.E[s]:
            if color[adj] == 'w':
                if self.checkRouteVisit(color, adj, d):
                    return True

        return False

        # time O(|V| + |E|)
        # space O(|V|)

    def checkRouteVisit(self, color, adj, d):
        color[adj] = 'g'

        if adj == d:
            return True

        for ver in self.E[adj]:
            if color[ver] == 'w':
                if self.checkRouteVisit(color, ver, d):
                    return True

        color[adj] = 'b'
        return False

    # -----------------------------------------------------------------------
    """
    4.7 Build Order: You are given a list of projects and a list of dependencies (which is a list of pairs of
    projects, where the second project is dependent on the first project). All of a project's dependencies
    must be built before the project is. Find a build order that will allow the projects to be built. If there
    is no valid build order, return an error.
    EXAMPLE
    Input:
    projects: a, b, c, d, e, f
    dependencies: (a, d), (f, b), (b, d), (f, a), (d, c)
    Output: f, e, a, b, d, c
    """

    def buildOrder(self, projects, dependencies):
        g = Graph()

        for d in dependencies:
            g.addEdge(d[0], d[1])
        res = []

        visit = {}
        for v in g.v:
            visit[v] = 'unvisited'

        for v in g.v:
            if visit[v] == 'unvisited':
                if self.recurBuildOrder(g, v, visit, res) == None:
                    return "Error - can't build order"

        for p in projects:
            if not p in g.v:
                res.append(p)

        res.reverse()

        return res

        # time O(|V| + |E|)
        # space O(|V| + |E|)

    def recurBuildOrder(self, g, v, visit, res):
        visit[v] = 'visiting'

        for ver in g.E[v]:
            if visit[ver] == 'unvisited':
                res = self.recurBuildOrder(g, ver, visit, res)
            elif visit[ver] == 'visiting':
                return None

        if res == None:
            return None

        res.append(v)
        visit[v] = 'visited'

        return res
