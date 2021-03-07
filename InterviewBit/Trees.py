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
