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
"""
Word Ladder II

Given two words (start and end), and a dictionary, find the shortest transformation sequence from start to end, such that:

Only one letter can be changed at a time
Each intermediate word must exist in the dictionary
If there are multiple such sequence of shortest length, return all of them. Refer to the example for more details.

Note:

All words have the same length.
All words contain only lowercase alphabetic characters.
Input Format

The first argument is string start.
The second argument is string end.
The third argument is an array of strings dict
Output Format

Return all transformation sequences such that first word of each sequence is start and last word is end, all intermediate words belongs to dictionary(dict) and consecutive words had atmost 1 difference.  
Example :

:

start = "hit"
end = "cog"
dict = ["hot","dot","dog","lot","log"]
Return

  [
    ["hit","hot","dot","dog","cog"],
    ["hit","hot","lot","log","cog"]
  ]
"""

from collections import defaultdict, deque


class Graph:
    def __init__(self):
        self.edges = defaultdict(set)
        self.vertices = set()

    def add(self, u, v):
        self.edges[u].add(v)
        self.edges[v].add(u)
        self.vertices.add(u)
        self.vertices.add(v)

    def get_paths(self, start, end):

        paths = []

        q = deque([(start, [])])
        visited = set()

        min_len = float('inf')

        while q:
            curr, path = q.popleft()

            if len(path) > min_len:
                continue

            path.append(curr)

            if curr == end:
                paths.append(path)
                min_len = min(min_len, len(path))
            else:
                for neigh in self.edges[curr]:
                    if neigh in visited:
                        continue
                    q.append((neigh, [v for v in path]))

            visited.add(curr)

        return [path for path in paths if len(path) == min_len]


class Solution:
    # @param start : string
    # @param end : string
    # @param dictV : list of strings
    # @return a list of list of strings
    def findLadders(self, start, end, dictV):
        graph = generate_graph(start, end, dictV)

        return graph.get_paths(start, end)

    # time O(n * L)
    # space O(n * L)


def generate_graph(start, end, words):
    words = set(words)

    idx_to_chars = generate_idx_to_char(words)

    graph = Graph()

    add_word_to_graph(graph, start, words, idx_to_chars)
    add_word_to_graph(graph, end, words, idx_to_chars)

    for word in words:
        add_word_to_graph(graph, word, words, idx_to_chars)

    return graph


def add_word_to_graph(graph, word, words, idx_to_chars):
    list_word = list(word)
    for i in range(len(word)):
        for c in idx_to_chars[i]:
            new_word = ''.join(list_word[:i] + [c] + list_word[i + 1:])
            if new_word in words and new_word != word:
                graph.add(word, new_word)


# for each idx, save all possible chars
def generate_idx_to_char(words):
    idx_to_chars = defaultdict(set)

    for word in words:
        for i, c in enumerate(word):
            idx_to_chars[i].add(c)

    return idx_to_chars


# -----------------------------------------------------------------------
"""
Largest Distance between nodes of a Tree

Problem Description

Given an arbitrary unweighted rooted tree which consists of N nodes.

The goal of the problem is to find largest distance between two nodes in a tree.

Distance between two nodes is a number of edges on a path between the nodes (there will be a unique path between any pair of nodes since it is a tree).

The nodes will be numbered 0 through N - 1.

The tree is given as an array A, there is an edge between nodes A[i] and i (0 <= i < N). Exactly one of the i's will have A[i] equal to -1, it will be root node.



Problem Constraints
1 <= N <= 40000



Input Format
First and only argument is an integer array A of size N.



Output Format
Return a single integer denoting the largest distance between two nodes in a tree.



Example Input
Input 1:

 A = [-1, 0, 0, 0, 3]


Example Output
Output 1:

 3


Example Explanation
Explanation 1:

 node 0 is the root and the whole tree looks like this: 
          0
       /  |  \
      1   2   3
               \
                4
"""

from collections import defaultdict, deque


class Graph:
    def __init__(self):
        self.edges = defaultdict(set)
        self.vertices = set()

    def add(self, u, v):

        self.edges[u].add(v)
        self.edges[v].add(u)
        self.vertices.add(u)
        self.vertices.add(v)

    def BFS(self, start):
        visited = set([start])
        q = deque([(start, 0)])
        visited.add(start)
        max_dist = 0
        max_node = None

        while q:
            curr, curr_dist = q.popleft()
            if curr_dist > max_dist:
                max_dist = curr_dist
                max_node = curr

            for adj in self.edges[curr]:
                if adj in visited:
                    continue
                visited.add(adj)
                q.append((adj, curr_dist + 1))

        return max_node, max_dist


class Solution:
    # @param A : list of integers
    # @return an integer
    def solve(self, A):

        graph = Graph()

        for i, v in enumerate(A):
            if i == 0:
                continue
            graph.add(v, i)

        node, max_dist = graph.BFS(0)

        node2, max_dist2 = graph.BFS(node)

        return max(max_dist, max_dist2)

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------

"""
Water Flow

Problem Description

Given an N x M matrix A of non-negative integers representing the height of each unit cell in a continent, the "Blue lake" touches the left and top edges of the matrix and the "Red lake" touches the right and bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

Find the number of cells from where water can flow to both the Blue and Red lake.



Problem Constraints
1 <= M, N <= 1000

1 <= A[i][j] <= 109



Input Format
First and only argument is a 2D matrix A.



Output Format
Return an integer denoting the number of cells from where water can flow to both the Blue and Red lake.



Example Input
Input 1:

 A = [
       [1, 2, 2, 3, 5]
       [3, 2, 3, 4, 4]
       [2, 4, 5, 3, 1]
       [6, 7, 1, 4, 5]
       [5, 1, 1, 2, 4]
     ]
Input 2:

 A = [
       [2, 2]
       [2, 2]
     ]


Example Output
Output 1:

 7
Output 2:

 4
"""

from collections import deque


class Solution:
    # @param A : list of list of integers
    # @return an integer
    # def solve(self, A):

    def solve(self, mat):

        n = len(mat)
        m = len(mat[0])
        blue_visited = set()
        red_visited = set()

        # first and last col
        for row in range(n):
            if (row, 0) not in blue_visited:
                bfs(mat, row, 0, blue_visited)

            if (row, m - 1) not in red_visited:
                bfs(mat, row, m - 1, red_visited)

        # first and last row
        for col in range(m):
            if (0, col) not in blue_visited:
                bfs(mat, 0, col, blue_visited)

            if (n - 1, col) not in red_visited:
                bfs(mat, n - 1, col, red_visited)

        return len(red_visited.intersection(blue_visited))

    # time O(n * m)
    # space O(n * m)


def bfs(mat, row, col, visited):
    if (row, col) in visited:
        return

    q = deque()
    q.append((row, col))
    visited.add((row, col))

    while q:
        row, col = q.popleft()
        for n_row, n_col in get_neighbors(row, col):
            if not is_valid(mat, n_row, n_col) or (n_row, n_col) in visited:
                continue

            if mat[row][col] <= mat[n_row][n_col]:
                q.append((n_row, n_col))
                visited.add((n_row, n_col))


def is_valid(mat, row, col):
    if row < 0 or col < 0 or row >= len(mat) or col >= len(mat[0]):
        return False

    return True


def get_neighbors(row, col):
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    res = []
    for dx, dy in dirs:
        res.append((row + dx, col + dy))

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
