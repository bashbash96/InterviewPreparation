"""
Given two strings s and t, return true if they are both one edit distance apart, otherwise return false.

A string s is said to be one distance apart from a string t if you can:

Insert exactly one character into s to get t.
Delete exactly one character from s to get t.
Replace exactly one character of s with a different character to get t.


Example 1:

Input: s = "ab", t = "acb"
Output: true
Explanation: We can insert 'c' into s to get t.
Example 2:

Input: s = "", t = ""
Output: false
Explanation: We cannot get t from s by only one step.
Example 3:

Input: s = "a", t = ""
Output: true
Example 4:

Input: s = "", t = "A"
Output: true


Constraints:

0 <= s.length <= 104
0 <= t.length <= 104
s and t consist of lower-case letters, upper-case letters and/or digits.
"""


class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        if abs(len(s) - len(t)) > 1:
            return False

        small, big = (s, t) if len(s) < len(t) else (t, s)

        for i in range(len(small)):

            if small[i] != big[i]:
                return small[i + 1:] == big[i + 1:] or small[i:] == big[i + 1:] or small[i + 1:] == big[i:]

        return len(small) == len(big) - 1

    # time O(n)
    # space O(n)


#         if not s and not t:
#             return False

#         if abs(len(s) - len(s)) > 1:
#             return False

#         return edit_dist(s, t, 0, 0, {})

#     # time O(max(n, m))
#     # space O(max(n, m))


def edit_dist(s, t, i1, i2, found_edit):
    if i1 == len(s) and i2 == len(t):
        return found_edit

    if i1 == len(s):
        if found_edit:
            return False

        return len(t) - i2 == 1

    if i2 == len(t):
        if found_edit:
            return False

        return len(s) - i1 == 1

    if s[i1] == t[i2]:
        return edit_dist(s, t, i1 + 1, i2 + 1, found_edit)

    if found_edit:
        return False

    return edit_dist(s, t, i1 + 1, i2 + 1, True) or edit_dist(s, t, i1, i2 + 1, True) or edit_dist(s, t, i1 + 1, i2,
                                                                                                   True)
