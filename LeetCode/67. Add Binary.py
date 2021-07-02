"""
Given two binary strings a and b, return their sum as a binary string.



Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"


Constraints:

1 <= a.length, b.length <= 104
a and b consist only of '0' or '1' characters.
Each string does not contain leading zeros except for the zero itself.
"""
BASE = 2


class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """

        a = list(reversed(a))
        b = list(reversed(b))

        return add_bin_numbers(a, b)

    # time O(a + b)
    # space O(a + b)


def add_bin_numbers(a, b):
    res = []

    p1, p2 = 0, 0
    carry, curr_sum = 0, 0

    while p1 < len(a) or p2 < len(b):
        val1 = a[p1] if p1 < len(a) else '0'
        val2 = b[p2] if p2 < len(b) else '0'

        curr_sum, carry = sum_two_vals(val1, val2, carry)
        res.append(str(curr_sum))
        p1 += 1
        p2 += 1

    if carry:
        res.append(str(carry))

    return ''.join(list(reversed(res)))


def sum_two_vals(val1, val2, carry):
    curr_sum = int(val1) + int(val2) + carry

    return curr_sum % BASE, curr_sum // BASE
