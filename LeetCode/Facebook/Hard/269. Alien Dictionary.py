"""
There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.

You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted lexicographically by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return "". If there are multiple solutions, return any of them.

A string s is lexicographically smaller than a string t if at the first letter where they differ, the letter in s comes before the letter in t in the alien language. If the first min(s.length, t.length) letters are the same, then s is smaller if and only if s.length < t.length.



Example 1:

Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
Example 2:

Input: words = ["z","x"]
Output: "zx"
Example 3:

Input: words = ["z","x","z"]
Output: ""
Explanation: The order is invalid, so return "".


Constraints:

1 <= words.length <= 100
1 <= words[i].length <= 100
words[i] consists of only lowercase English letters.
"""

from collections import defaultdict, deque


class Solution(object):
    def alienOrder(self, words):
        """
        :type words: List[str]
        :rtype: str
        """

        graph, letters = build_graph(words)
        if graph == '':
            return ''

        status = defaultdict()
        res = []

        for l in letters:
            status[l] = 'not visited'

        for l in letters:
            if status[l] == 'not visited':
                if not dfs(graph, l, status, res):
                    return ''

        return ''.join(res[::-1])

    # time O(E + V)
    # space O(1)


def dfs(graph, v, status, res):
    status[v] = 'visiting'

    for adj in graph[v]:
        if status[adj] == 'visiting':
            return False

        if status[adj] == 'not visited':
            if not dfs(graph, adj, status, res):
                return False

    status[v] = 'visited'
    res.append(v)
    return True


def build_graph(words):
    letters = set()
    graph = defaultdict(set)

    for i in range(len(words) - 1):

        a, b = get_dependency(words[i], words[i + 1])

        if a == -2:  # invalid order
            return '', ''

        if a == -1:  # didn't find any helpful order
            continue

        if b not in graph[a]:  # a comes before b
            graph[a].add(b)

        letters = letters.union(set(words[i]))

    letters = letters.union(set(words[-1]))

    return graph, letters


def get_dependency(word1, word2):
    i, j = 0, 0

    while i < len(word1) and j < len(word2):
        if word1[i] != word2[j]:
            return word1[i], word2[j]

        i += 1
        j += 1

    if len(word2) < len(word1):
        return -2, -2
    return -1, -1


"""
["wrt","wrf","er","ett","rftt"]

t : [f, ]
w: [e, r, ]
r: [t, ]
e: [r, ]



"""
