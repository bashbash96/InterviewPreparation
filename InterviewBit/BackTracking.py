# -----------------------------------------------------------------------
"""
A

ll Unique Permutations REVISE

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
