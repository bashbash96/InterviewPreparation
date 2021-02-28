# -----------------------------------------------------------------------
"""
Search for a Range

Given a sorted array of integers A(0 based index) of size N, find the starting and ending position of a given integar B in array A.

Your algorithm’s runtime complexity must be in the order of O(log n).

Return an array of size 2, such that first element = starting position of B in A and second element = ending position of B in A, if B is not found in A return [-1, -1].



Input Format

The first argument given is the integer array A.
The second argument given is the integer B.
Output Format

 Return an array of size 2, such that first element = starting position of B in A and second element = ending position of B in A, if B is not found in A return [-1, -1].
Constraints

1 <= N <= 10^6
1 <= A[i], B <= 10^9
For Example

Input 1:
    A = [5, 7, 7, 8, 8, 10]
    B = 8
Output 1:
    [3, 4]
Explanation 1:
    First occurence of 8 in A is at index 3
    Second occurence of 8 in A is at index 4
    ans = [3, 4]

Input 2:
    A = [5, 17, 100, 111]
    B = 3
Output 2:
    [-1, -1]
"""


class Solution:
    # @param A : tuple of integers
    # @param B : integer
    # @return a list of integers
    def searchRange(self, A, B):
        left = get_limit(A, B, 'left')
        right = get_limit(A, B, 'right')

        return [left, right]

    # time O(log(n))
    # space O(1)


def get_limit(arr, num, limit):
    start, end = 0, len(arr) - 1
    res = -1
    while start <= end:
        mid = (start + end) // 2
        if num < arr[mid]:
            end = mid - 1
        elif num > arr[mid]:
            start = mid + 1
        else:
            res = mid
            if limit == 'left':
                end = mid - 1
            else:
                start = mid + 1

    return res


# -----------------------------------------------------------------------
"""
Implement Power Function

Implement pow(x, n) % d.

In other words, given x, n and d,

find (xn % d)

Note that remainders on division cannot be negative.
In other words, make sure the answer you return is non negative.

Input : x = 2, n = 3, d = 3
Output : 2

2^3 % 3 = 8 % 3 = 2.
"""


class Solution:
    # @param x : integer
    # @param n : integer
    # @param d : integer
    # @return an integer
    def pow(self, x, n, d):

        if n == 0:
            if x == 0:
                return 0
            return 1

        if d == 1:
            return 0

        calc = self.pow(x, n // 2, d)
        if n % 2 == 0:
            return (calc * calc) % d
        else:
            return (x * calc * calc) % d

    # time O(log(n))
    # space O(log(n))

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
