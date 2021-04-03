# -----------------------------------------------------------------------
"""
17.1 Add Without Plus: Write a function that adds two numbers. You should not use + or any arithmetic
operators.
"""


def add(num1, num2):
    if num2 == 0:
        return num1

    curr_sum = num1 ^ num2
    carry = (num1 & num2) << 1

    return add(curr_sum, carry)

    # time O(log(n))
    # space O(log(n))


# -----------------------------------------------------------------------
"""
17.2 Shuffle: Write a method to shuffle a deck of cards. It must be a perfect shuffle-in other words, each
of the 52! permutations of the deck has to be equally likely. Assume that you are given a random
number generator which is perfect.
"""
from random import randint


def shuffle(cards):
    for i in range(len(cards) - 1, -1, -1):
        r = randint(0, i)
        cards[i], cards[r] = cards[r], cards[i]

    return cards

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
17.3 Random Set: Write a method to randomly generate a set of m integers from an array of size n. Each
element must have equal probability of being chosen.
"""


def random_set(arr, m):
    res = set()
    for i in range(len(arr) - 1, -1, -1):
        r = randint(0, i)
        res.add(arr[r])
        arr[r], arr[i] = arr[i], arr[r]
        if len(res) == m:
            break

    return res

    # time O(m)
    # space O(m)


def random_set_reservoir(arr, m):
    res = arr[:m]

    for i in range(m, len(arr)):
        r = randint(0, i - 1)
        if r < m:
            res[r] = arr[i]

    return res

    # time O(n)
    # space O(m)


# -----------------------------------------------------------------------
"""
17.4 Missing Number: An array A contains all the integers from 0 to n, except for one number which
is missing. In this problem, we cannot access an entire integer in A with a single operation. The
elements of A are represented in binary, and the only operation we can use to access them is "fetch
the jth bit of A[i]," which takes constant time. Write code to find the missing integer. Can you do it
in O(n) time?
"""


def missing_number(arr):
    res = 0

    for i in range(64):
        curr = 0
        opp_num = 0
        for num in arr:
            curr ^= get_bit(num, i)
            curr ^= get_bit(opp_num, i)
            opp_num += 1

        curr ^= get_bit(opp_num, i)
        res = res | (curr << i)

    return res

    # time O(n)
    # space O(1)


def get_bit(num, i):
    res = 1
    return ((res << i) & num) >> i


# -----------------------------------------------------------------------
"""
17.5 Letters and Numbers: Given an array filled with letters and numbers, find the longest subarray with
an equal number of letters and numbers.
"""

from collections import defaultdict


def longest_sub(arr):
    count_letters = 0
    count_numbers = 0

    max_len = 0
    max_left = 0
    seen = defaultdict()
    seen[0] = 0

    for i, num in enumerate(arr):
        if num.isnumeric():
            count_numbers += 1
        if num.isalpha():
            count_letters += 1

        diff = count_numbers - count_letters
        if diff in seen:
            new_len = i - seen[diff]
            if new_len > max_len:
                max_len = new_len
                max_left = seen[diff] + 1
        else:
            seen[diff] = i

    return arr[max_left: max_left + max_len]


# -----------------------------------------------------------------------
"""
17.7 Baby Names: Each year, the government releases a list of the 10,000 most common baby names
and their frequencies (the number of babies with that name). The only problem with this is that
some names have multiple spellings. For example, "John" and "Jon" are essentially the same name
but would be listed separately in the list. Given two lists, one of names/frequencies and the other
of pairs of equivalent names, write an algorithm to print a new list of the true frequency of each
name. Note that if John and Jon are synonyms, and Jon and Johnny are synonyms, then John and
Johnny are synonyms. (It is both transitive and symmetric.) In the final list, any name can be used
as the "real" name.
EXAMPLE
Input:
Names: John (15), Jon (12), Chris (13), Kris (4), Christopher (19)
Synonyms: (Jon, John), (John, Johnny), (Chris, Kris), (Chris, Christopher)
Output: John (27), Kris (36)
"""


def baby_names(name_freq, name_syn):
    graph = build_graph(name_syn)

    visited = set()
    res = []
    for name in name_freq:
        if name not in visited:
            curr = dfs_visit(graph, name_freq, name, visited)
            res.append((name, curr))

    return res

    # time O(n + m)
    # space (m)


def dfs_visit(graph, name_freq, name, visited):
    visited.add(name)

    curr = name_freq.get(name, 0)
    for adj in graph[name]:
        if adj not in visited:
            curr += dfs_visit(graph, name_freq, adj, visited)

    return curr


def build_graph(name_syn):
    graph = defaultdict(set)

    for name1, name2 in name_syn:
        graph[name1].add(name2)
        graph[name2].add(name1)

    return graph


# freq = {'john': 15, 'jon': 12, 'chris': 13, 'kris': 4, 'christopher': 19}
# syn = [('john', 'jon'), ('john', 'johnny'), ('chris', 'kris'), ('chris', 'christopher')]
# print(baby_names(freq, syn))

# -----------------------------------------------------------------------
"""
17.8 Circus Tower: A circus is designing a tower routine consisting of people standing atop one another's
shoulders. For practical and aesthetic reasons, each person must be both shorter and lighter than
the person below him or her. Given the heights and weights of each person in the circus, write a
method to compute the largest possible number of people in such a tower.
"""


def longest_tower(pairs):
    pairs.sort()
    n = len(pairs)

    sequence_length = [1] * n
    max_sequence = 0

    for i in range(1, n):
        curr_max = 0
        for j in range(i):
            if can_stand_above(pairs[j], pairs[i]):
                curr_max = max(curr_max, sequence_length[j])

        sequence_length[i] = curr_max + 1
        max_sequence = max(max_sequence, sequence_length[i])

    return max_sequence

    # time O(n^2)
    # space O(n)


def can_stand_above(pair1, pair2):
    return pair2[0] > pair1[0] and pair2[1] > pair1[1]


# -----------------------------------------------------------------------
"""
17.9 Kth Multiple: Design an algorithm to find the kth number such that the only prime factors are 3, 5,
and 7. Note that 3,5, and 7 do not have to be factors, but it should not have any other prime factors.
For example, the first several multiples would be (in order) 1,3, 5, 7, 9, 15,21.
"""


def kth_element(k):
    if k < 0:
        return 0

    vals = [1]
    idx = 0
    mult_idx = 0
    mult = [3, 5, 7]
    seen = set([1])
    while len(vals) < k:

        if mult_idx == len(mult):
            mult_idx = 0
            idx += 1
        curr_val = mult[mult_idx] * vals[idx]
        if curr_val in seen:
            mult_idx += 1
            continue
        vals.append(curr_val)
        seen.add(curr_val)
        mult_idx += 1

    return vals[-1]

    # time O(k)
    # space O(k)


# -----------------------------------------------------------------------
"""
17.10 Majority Element: A majority element is an element that makes up more than half of the items in
an array. Given a positive integers array, find the majority element. If there is no majority element,
return -1. Do this in O( N) time and O( 1) space.
Input:
Output:
SOLUTION
1 2 5 9 5 9 5 5 5
5
"""


def majority(arr):
    val, count = None, 0

    for num in arr:
        if count == 0:
            val = num
            count = 1
        elif val == num:
            count += 1
        else:
            count -= 1

    return val

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
17.11 Word Distance: You have a large text file containing words. Given any two words, find the shortest
distance (in terms of number of words) between them in the file. If the operation will be repeated
many times for the same file (but different pairs of words), can you optimize your solution?
"""


def smallest_distance(word1, word2, words):
    locations = defaultdict()

    smallest = float('inf')

    for i, word in enumerate(words):

        if word == word1:
            locations[word] = i
            if len(locations) > 1:
                diff = abs(i - locations[word2])
                if diff < smallest:
                    smallest = abs(i - diff)
        elif word == word2:
            locations[word] = i
            if len(locations) > 1:
                diff = abs(i - locations[word1])
                if diff < smallest:
                    smallest = abs(i - diff)

    return smallest


"""
if there are a lot of calls for this function, we can store indexes for each word
then for each pair, we can get the least difference between two sorted lists of indexes.
"""

# -----------------------------------------------------------------------
"""
17.12 BiNode: Consider a simple data structure called BiNode, which has pointers to two other nodes. The
data structure BiNode could be used to represent both a binary tree (where nodel is the left node
and node2 is the right node) or a doubly linked list (where nodel is the previous node and node2
is the next node). Implement a method to convert a binary search tree (implemented with BiNode)
into a doubly linked list. The values should be kept in order and the operation should be performed
in place (that is, on the original data structure).
"""


def convert_BST_to_LL(root):
    if not root:
        return root

    left = convert_BST_to_LL(root.left)
    right = convert_BST_to_LL(root.right)
    root.left = root.right = root

    if not left and not right:
        return root

    if not left:
        return merge(root, right)

    if not right:
        return merge(left, root)

    return merge(merge(left, root), right)

    # time O(n)
    # space O(1)


def merge(left, right):
    last_left = left.left
    last_right = right.left

    last_left.right = right
    right.left = last_left

    last_right.right = left
    left.left = last_right

    return left


# -----------------------------------------------------------------------
"""
17.13 Re-Space: Oh, no! You have accidentally removed all spaces, punctuation, and capitalization in a
lengthy document. A sentence like "I reset the computer. It still didn J t boot!"
became"iresetthecomputeritstilldidntboot': You'll deal with the punctuation and capitalization
later; right now you need to re-insert the spaces. Most of the words are in a dictionary but
a few are not. Given a dictionary (a list of strings) and the document (a string), design an algorithm
to unconcatenate the document in a way that minimizes the number of unrecognized characters.
EXAMPLE
Input: jesslookedjustliketimherbrother
Output: jess looked just like tim her brother (7 unrecognized characters)


 s = jesslookedjustliketimherbrother
 words = {...}

if s[i:] in words:
    return 0
if i >= len(s):
return 0, ''
 min_undefined_chars(i,words) = min(min_undefined_chars(i + 1, words) + 1,
                                        for k in range(i, end):
                                            if s[i:k] in words:
                                                min_undefined_chars(k, words))
 
"""


def get_min(s, words):
    return min_undefined_chars(s, 0, words, {})


def min_undefined_chars(s, i, words, memo):
    if i >= len(s):
        return 0, ''

    if s[i:] in words:
        return 0, s[i:]

    if i in memo:
        return memo[i]

    curr, curr_str = min_undefined_chars(s, i + 1, words, memo)
    curr += 1
    curr_str = s[i] + curr_str
    for end in range(i + 1, len(s) + 1):
        if s[i:end] in words:
            count, string = min_undefined_chars(s, end, words, memo)
            if count < curr:
                curr = count
                curr_str = ' ' + s[i:end] + ' ' + string

    memo[i] = curr, curr_str

    return memo[i]


# -----------------------------------------------------------------------
"""
17.14 Smallest K: Design an algorithm to find the smallest K numbers in an array.
"""
import heapq


def smallest_k(arr, k):
    min_k = []

    for val in arr:
        heapq.heappush(min_k, -val)
        if len(min_k) > k:
            heapq.heappop(min_k)

    return [-val for val in min_k]

    # time O(n * log(k))
    # space O(k)

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
