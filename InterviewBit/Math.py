# -----------------------------------------------------------------------
"""
Grid Unique Paths

A robot is located at the top-left corner of an A x B grid (marked ‘Start’ in the diagram below).

Path Sum: Example 1

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked ‘Finish’ in the diagram below).

How many possible unique paths are there?
"""


class Solution:
    # @param A : integer
    # @param B : integer
    # @return an integer
    def uniquePaths(self, A, B):
        memo = [[0 for j in range(B)] for i in range(A)]
        for row in range(A):
            memo[row][0] = 1

        for col in range(B):
            memo[0][col] = 1

        for row in range(1, A):
            for col in range(1, B):
                memo[row][col] = memo[row - 1][col] + memo[row][col - 1]

        return memo[A - 1][B - 1]

    # time O(A * B)
    # space O(A * B)


# -----------------------------------------------------------------------
"""
Greatest Common Divisor

Given 2 non negative integers m and n, find gcd(m, n)

GCD of 2 integers m and n is defined as the greatest integer g such that g is a divisor of both m and n.
Both m and n fit in a 32 bit signed integer.

Example

m : 6
n : 9

GCD(m, n) : 3 
"""


class Solution:
    # @param A : integer
    # @param B : integer
    # @return an integer
    def gcd(self, A, B):
        while B != 0:
            A, B = B, A % B

        return A

    # time O(log(n))
    # space O(1)


# -----------------------------------------------------------------------
"""
Sum of pairwise Hamming Distance

Hamming distance between two non-negative integers is defined as the number of positions at which the corresponding bits are different.

For example,

HammingDistance(2, 7) = 2, as only the first and the third bit differs in the binary representation of 2 (010) and 7 (111).

Given an array of N non-negative integers, find the sum of hamming distances of all pairs of integers in the array.
Return the answer modulo 1000000007.

Example

Let f(x, y) be the hamming distance defined above.

A=[2, 4, 6]

We return,
f(2, 2) + f(2, 4) + f(2, 6) + 
f(4, 2) + f(4, 4) + f(4, 6) +
f(6, 2) + f(6, 4) + f(6, 6) = 

0 + 2 + 1
2 + 0 + 1
1 + 1 + 0 = 8
"""

NUM_OF_BITS = 32
MOD = pow(10, 9) + 7


class Solution:
    # @param A : tuple of integers
    # @return an integer
    def hammingDistance(self, A):
        res = 0

        for i in range(NUM_OF_BITS):

            one_count = 0
            for num in A:
                num >>= i
                if num & 1:
                    one_count += 1
            zero_count = len(A) - one_count

            res = res + (one_count * zero_count * 2) % MOD

        return res % MOD

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
Problem Description

Given a numeric string A representing a large number you need to find the next smallest palindrome greater than this number.



Problem Constraints
1 <= |A| <= 100

A doesn't start with zeroes and always contain digits from 0-9.
"""


class Solution:
    # @param A : string
    # @return a strings
    def solve(self, A):
        n = len(A)
        num = [int(c) for c in A]

        mid = int(n / 2)

        # A bool variable to check if copy of left
        # side to right is sufficient or not
        leftsmaller = False

        # end of left side is always 'mid -1'
        i = mid - 1

        # Beginning of right side depends
        # if n is odd or even
        j = mid + 1 if (n % 2) else mid

        # Initially, ignore the middle same digits
        while (i >= 0 and num[i] == num[j]):
            i -= 1
            j += 1

        # Find if the middle digit(s) need to be
        # incremented or not (or copying left
        # side is not sufficient)
        if (i < 0 or num[i] < num[j]):
            leftsmaller = True

        # Copy the mirror of left to tight
        while (i >= 0):
            num[j] = num[i]
            j += 1
            i -= 1

        # Handle the case where middle
        # digit(s) must be incremented.
        # This part of code is for CASE 1 and CASE 2.2
        if (leftsmaller == True):

            carry = 1
            i = mid - 1

            # If there are odd digits, then increment
            # the middle digit and store the carry
            if (n % 2 == 1):

                num[mid] += carry
                carry = num[mid] // 10
                num[mid] %= 10
                j = mid + 1

            else:
                j = mid

                # Add 1 to the rightmost digit of the
            # left side, propagate the carry
            # towards MSB digit and simultaneously
            # copying mirror of the left side
            # to the right side.
            while (i >= 0 and carry):
                num[i] += carry
                carry = num[i] // 10
                num[i] %= 10
                num[j] = num[i]  # copy mirror to right
                j += 1
                i -= 1

            if carry:
                num[0] = 1
                num.append(1)

        return ''.join([str(c) for c in num])

    # time O(n)
    # space O(n)

# -----------------------------------------------------------------------
