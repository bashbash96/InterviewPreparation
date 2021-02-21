# -----------------------------------------------------------------------
"""
Single Number II

Given an array of integers, every element appears thrice except for one which occurs once.

Find that element which does not appear thrice.

Note: Your algorithm should have a linear runtime complexity.

Could you implement it without using extra memory?

Input Format:

    First and only argument of input contains an integer array A
Output Format:

    return a single integer.

Constraints:

2 <= N <= 5 000 000
0 <= A[i] <= INT_MAX
For Examples :

Example Input 1:
    A = [1, 2, 4, 3, 3, 2, 2, 3, 1, 1]
Example Output 1:
    4
Explanation:
    4 occur exactly once
Example Input 2:
    A = [0, 0, 0, 1]
Example Output 2:
    1
"""

from collections import Counter


class Solution:
    # @param A : tuple of integers
    # @return an integer
    # def singleNumber(self, A):
    #     counts = Counter(A)
    #     for num in counts:
    #         if counts[num] == 1:
    #             return num

    # time O(n)
    # space O(n)

    def singleNumber(self, A):
        res = 0

        for i in range(32):
            curr_sum = 0
            for num in A:
                if (num >> i) & 1:
                    curr_sum += 1

            res = res | ((curr_sum % 3) << i)

        return res

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
Different Bits Sum Pairwise

We define f(X, Y) as number of different corresponding bits in binary representation of X and Y. For example, f(2, 7) = 2, since binary representation of 2 and 7 are 010 and 111, respectively. The first and the third bit differ, so f(2, 7) = 2.

You are given an array of N positive integers, A1, A2 ,…, AN. Find sum of f(Ai, Aj) for all pairs (i, j) such that 1 ≤ i, j ≤ N. Return the answer modulo 109+7.

For example,

A=[1, 3, 5]

We return

f(1, 1) + f(1, 3) + f(1, 5) + 
f(3, 1) + f(3, 3) + f(3, 5) +
f(5, 1) + f(5, 3) + f(5, 5) =

0 + 1 + 1 +
1 + 0 + 2 +
1 + 2 + 0 = 8
"""

BITS_NUM = 32
MOD = pow(10, 9) + 7


class Solution:
    # @param A : list of integers
    # @return an integer
    def cntBits(self, A):
        total_count = 0

        for i in range(BITS_NUM):
            count_ones, count_zeros = 0, 0
            for num in A:
                num >>= i
                if num & 1:
                    count_ones += 1
                else:
                    count_zeros += 1

            total_count += (count_ones * count_zeros * 2) % MOD

        return total_count % MOD

    # time O(n)
    # space O(1)

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
# -----------------------------------------------------------------------
