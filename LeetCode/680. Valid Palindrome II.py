"""
Given a string s, return true if the s can be palindrome after deleting at most one character from it.



Example 1:

Input: s = "aba"
Output: true
Example 2:

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.
Example 3:

Input: s = "abc"
Output: false


Constraints:

1 <= s.length <= 105
s consists of lowercase English letters.
"""


class Solution(object):
    def validPalindrome(self, s):

        """
        :type s: str
        :rtype: bool
        """

        left, right = 0, len(s) - 1

        while left < right:
            if s[left] != s[right]:
                op1, op2 = s[left:right], s[left + 1: right + 1]

                return op1 == op1[::-1] or op2 == op2[::-1]

            left += 1
            right -= 1

        return True

        # return is_valid_pal(s, 0, len(s) - 1, 0)

    # time O(n)
    # space O(n)


def is_valid_pal(s, left, right, count):
    if count > 1:
        return False

    if left >= right:
        return True

    if s[left] != s[right]:
        return is_valid_pal(s, left + 1, right, count + 1) or is_valid_pal(s, left, right - 1, count + 1)

    return is_valid_pal(s, left + 1, right - 1, count)


"""


with ordering chars
1. all chars are even count.
2. all chars arre even count except one.
3. all chars are even count except two.


without ordering
if there is a mismatch between lef and right, exclude one of them.


"""
