# -----------------------------------------------------------------------
"""
Least Common Ancestor

Find the lowest common ancestor in an unordered binary tree given two values in the tree.

 Lowest common ancestor : the lowest common ancestor (LCA) of two nodes v and w in a tree or directed acyclic graph (DAG) is the lowest (i.e. deepest) node that has both v and w as descendants.
Example :


        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2_     0        8
         /   \
         7    4
For the above tree, the LCA of nodes 5 and 1 is 3.

 LCA = Lowest common ancestor
Please note that LCA for nodes 5 and 4 is 5.

You are given 2 values. Find the lowest common ancestor of the two nodes represented by val1 and val2
No guarantee that val1 and val2 exist in the tree. If one value doesn’t exist in the tree then return -1.
There are no duplicate values.
You can use extra memory, helper functions, and can modify the node struct but, you can’t add a parent pointer.
"""


# Definition for a  binary tree node
# class TreeNode:
#    def __init__(self, x):
#        self.val = x
#        self.left = None
#        self.right = None

class Solution:
    # @param A : root node of tree
    # @param B : integer
    # @param C : integer
    # @return an integer
    def lca(self, A, B, C):

        path1, path2 = [], []

        val1_exist, val2_exist = get_path(A, path1, B), get_path(A, path2, C)

        if not val1_exist or not val2_exist:
            return -1

        i = 0
        while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
            i += 1

        return path1[i - 1]

    # time O(n)
    # space O(n)


def get_path(node, path, val):
    if not node:
        return False

    path.append(node.val)

    if node.val == val:
        return True

    left = get_path(node.left, path, val)

    if left:
        return True

    right = get_path(node.right, path, val)

    if right:
        return True

    path.pop()

    return False


# -----------------------------------------------------------------------
"""
Sum Root to Leaf Numbers

Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers % 1003.

Example :

    1
   / \
  2   3
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.

Return the sum = (12 + 13) % 1003 = 25 % 1003 = 25.
"""


# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param A : root node of tree
    # @return an integer

    def __init__(self):
        self.res = 0

    def sumNumbers(self, A):

        path = []
        self.calc_path_nums(A, path)

        return self.res % 1003

    # time O(n)
    # space O(h)

    def calc_path_nums(self, node, path):
        if not node:
            return

        if not node.left and not node.right:
            self.res += get_num(path + [node.val])
            return

        path.append(node.val)

        self.calc_path_nums(node.left, path)
        self.calc_path_nums(node.right, path)
        path.pop()


def get_num(digs):
    res = 0

    for dig in digs:
        res = res * 10 + dig

    return res


# -----------------------------------------------------------------------
"""
Order of People Heights

You are given the following :

A positive number N
Heights : A list of heights of N persons standing in a queue
Infronts : A list of numbers corresponding to each person (P) that gives the number of persons who are taller than P and standing in front of P
You need to return list of actual order of persons’s height

Consider that heights will be unique

Example

Input : 
Heights: 5 3 2 6 1 4
InFronts: 0 1 2 0 3 2
Output : 
actual order is: 5 3 2 1 6 4 
"""


class Solution:
    # @param A : list of integers
    # @param B : list of integers
    # @return a list of integers
    def order(self, A, B):
        pairs = [(val, count) for val, count in zip(A, B)]
        pairs.sort()
        n = len(pairs)

        res = [None for _ in range(n)]

        for pair in pairs:
            idx = get_appropriate_idx(res, pair)
            res[idx] = pair[0]

        return res

    # time O(n^2)
    # space O(n)


def get_appropriate_idx(res, pair):
    val, count = pair
    idx = 0

    while idx < len(res) and count > 0:
        if res[idx] == None:
            count -= 1
        idx += 1

    while idx < len(res) and res[idx] != None and res[idx] <= val:
        idx += 1

    return idx


# -----------------------------------------------------------------------
"""
Shortest Unique Prefix

Find shortest unique prefix to represent each word in the list.

Example:

Input: [zebra, dog, duck, dove]
Output: {z, dog, du, dov}
where we can see that
zebra = z
dog = dog
duck = du
dove = dov
"""


class Trie:
    def __init__(self):
        self.head = {}

    def insert(self, word):
        curr = self.head

        for char in word:
            if char not in curr:
                curr[char] = {}
            curr = curr[char]

        curr['*'] = '*'

    def shortest_prefix(self, word):

        if word[0] not in self.head:
            raise ValueError("Invalid word")
        res = []
        self.shortes_pre(word, 0, res, self.head[word[0]])
        if res == []:
            return word[0]
        return ''.join(res)

    def shortes_pre(self, word, idx, res, curr):

        if idx >= len(word) - 1:
            return False

        res.append(word[idx])
        found = self.shortes_pre(word, idx + 1, res, curr[word[idx + 1]])
        if found:
            return True

        if len(curr) > 1:
            res.append(word[idx + 1])
            return True

        res.pop()
        return False


class Solution:
    # @param A : list of strings
    # @return a list of strings
    def prefix(self, A):

        tree = Trie()

        for word in A:
            tree.insert(word)

        res = []
        for word in A:
            res.append(tree.shortest_prefix(word))

        return res

    # time O(n * l)
    # space O(n * l)


# -----------------------------------------------------------------------
"""
Invert the Binary Tree

Given a binary tree, invert the binary tree and return it.
Look at the example for more details.

Example :
Given binary tree

     1
   /   \
  2     3
 / \   / \
4   5 6   7
invert and return

     1
   /   \
  3     2
 / \   / \
7   6 5   4
"""


# Definition for a  binary tree node
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param A : root node of tree
    # @return the root node in the tree
    def invertTree(self, node):
        if not node:
            return node

        left = self.invertTree(node.left)
        right = self.invertTree(node.right)

        node.left = right
        node.right = left

        return node

    # time O(n)
    # space O(h)

# -----------------------------------------------------------------------
