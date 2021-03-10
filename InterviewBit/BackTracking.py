# -----------------------------------------------------------------------
"""
A

ll Unique Permutations

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

Example :
[1,1,2] have the following unique permutations:

[1,1,2]
[1,2,1]
[2,1,1]


NOTE : No 2 entries in the permutation sequence should be the same.
"""

from collections import Counter


class Solution:
    def permuteUnique(self, nums):
        result = []

        def generate_perms(curr_perm, counter):
            if len(curr_perm) == len(nums):
                result.append(curr_perm[::])
                return

            for num in counter:
                if counter[num] > 0:
                    curr_perm.append(num)
                    counter[num] -= 1

                    generate_perms(curr_perm, counter)
                    curr_perm.pop()
                    counter[num] += 1

        generate_perms([], Counter(nums))

        return result

    # time O(n! * n)
    # space O(n! * n)


# sol2
class Solution:
    # @param A : list of integers
    # @return a list of list of integers
    def permute(self, A):
        prev_perm = [[]]
        for num in A:
            curr_perm = generate_perm(prev_perm, num)
            prev_perm = curr_perm[::]

        return curr_perm

    # time O(n! * n^2)
    # space O(n! * n)


def generate_perm(prev, num):
    curr = []
    seen = set()
    for perm in prev:
        for i in range(len(perm) + 1):
            curr_perm = perm[:i] + [num] + perm[i:]
            perm_id = ",".join([str(val) for val in curr_perm])
            if perm_id in seen:
                continue
            seen.add(perm_id)
            curr.append(curr_perm)

    return curr


# -----------------------------------------------------------------------
"""
Permutations

Given a collection of numbers, return all possible permutations.

Example:

[1,2,3] will have the following permutations:

[1,2,3]
[1,3,2]
[2,1,3] 
[2,3,1] 
[3,1,2] 
[3,2,1]
 NOTE
No two entries in the permutation sequence should be the same.
For the purpose of this problem, assume that all the numbers in the collection are unique.
"""


class Solution:
    # @param A : list of integers
    # @return a list of list of integers
    def permute(self, A):

        if len(A) == 0:
            return []

        prev_perm = [[A[0]]]
        for i in range(1, len(A)):
            new_num = A[i]
            prev_perm = generate_perm(prev_perm, new_num)

        return prev_perm

    # time O(n! * n)
    # space O(n!)


def generate_perm(prev_perm, num):
    res = []
    for perm in prev_perm:
        for i in range(len(perm) + 1):
            res.append(perm[:i] + [num] + perm[i:])

    return res


# -----------------------------------------------------------------------
"""
Palindrome Partitioning

Given a string s, partition s such that every string of the partition is a palindrome.

Return all possible palindrome partitioning of s.

For example, given s = "aab",
Return

  [
    ["a","a","b"]
    ["aa","b"],
  ]
 Ordering the results in the answer : Entry i will come before Entry j if :
len(Entryi[0]) < len(Entryj[0]) OR
(len(Entryi[0]) == len(Entryj[0]) AND len(Entryi[1]) < len(Entryj[1])) OR
*
*
*
(len(Entryi[0]) == len(Entryj[0]) AND … len(Entryi[k] < len(Entryj[k]))
"""


class Solution:
    # @param A : string
    # @return a list of list of strings
    def partition(self, A):
        res = []

        recur_partition(A, 0, [], res)

        return res

    # time O(n^k * n)
    # space O(n)


def recur_partition(string, idx, curr, res):
    if idx >= len(string):
        res.append(curr[:])
        return

    for i in range(idx, len(string)):
        if is_palindrome(string, idx, i):
            recur_partition(string, i + 1, curr + [string[idx: i + 1]], res)


def is_palindrome(string, i, j):
    while i < j and string[i] == string[j]:
        i += 1
        j -= 1

    return i >= j

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
