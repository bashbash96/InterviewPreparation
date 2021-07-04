"""
A valid number can be split up into these components (in order):

A decimal number or an integer.
(Optional) An 'e' or 'E', followed by an integer.
A decimal number can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One of the following formats:
One or more digits, followed by a dot '.'.
One or more digits, followed by a dot '.', followed by one or more digits.
A dot '.', followed by one or more digits.
An integer can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One or more digits.
For example, all the following are valid numbers: ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"], while the following are not valid numbers: ["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"].

Given a string s, return true if s is a valid number.



Example 1:

Input: s = "0"
Output: true
Example 2:

Input: s = "e"
Output: false
Example 3:

Input: s = "."
Output: false
Example 4:

Input: s = ".1"
Output: true


Constraints:

1 <= s.length <= 20
s consists of only English letters (both uppercase and lowercase), digits (0-9), plus '+', minus '-', or dot '.'.
"""


class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """

        found_digit = found_dot = found_expo = False

        for i, c in enumerate(s):

            if c.isdigit():
                found_digit = True
            elif is_sign(s, i):
                if i > 0 and not is_exponential(s, i - 1):
                    return False
            elif is_exponential(s, i):
                if found_expo or not found_digit:
                    return False
                found_expo = True
                found_digit = False
            elif is_dot(s, i):
                if found_dot or found_expo:
                    return False
                found_dot = True
            else:
                return False

        return found_digit

        # time O(n)
    # space O(1)


def is_sign(s, idx):
    return s[idx] == '-' or s[idx] == '+'


def is_dot(s, idx):
    return s[idx] == '.'


def is_exponential(s, idx):
    return s[idx] == 'e' or s[idx] == 'E'
