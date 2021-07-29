"""
Given an integer n, return all the strobogrammatic numbers that are of length n. You may return the answer in any order.

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).



Example 1:

Input: n = 2
Output: ["11","69","88","96"]
Example 2:

Input: n = 1
Output: ["0","1","8"]


Constraints:

1 <= n <= 14
"""

opposite = {
    '1': '1',
    '6': '9',
    '9': '6',
    '8': '8',
    '0': '0'
}


class Solution(object):
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

        res = []
        generate_stro_numbers([None] * n, 0, n - 1, res)

        return res

    # time O(n * 5^n)


def generate_stro_numbers(curr, start, end, res):
    if start > end:
        res.append(''.join(curr))
        return

    for num in opposite:
        if start == end and num in ('6', '9'):
            continue

        if start != end and start == 0 and num == '0':
            continue

        curr[start], curr[end] = num, opposite[num]

        generate_stro_numbers(curr, start + 1, end - 1, res)


"""
1 6 8 9 0

1 -> 1, 8, 0
2 -> 11, 88, 69, 96
3-> 111, 101, 181, ...
"""
