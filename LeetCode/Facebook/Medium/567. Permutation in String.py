"""
Given two strings s1 and s2, return true if s2 contains the permutation of s1.

In other words, one of s1's permutations is the substring of s2.



Example 1:

Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").
Example 2:

Input: s1 = "ab", s2 = "eidboaoo"
Output: false


Constraints:

1 <= s1.length, s2.length <= 104
s1 and s2 consist of lowercase English letters.
"""

from collections import Counter


class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """

        if len(s1) > len(s2):
            return False

        s1_count = Counter(s1)
        s2_count = Counter(s2[:len(s1)])

        left = 0
        for right in range(len(s1), len(s2)):
            if s1_count == s2_count:
                return True

            prev_char = s2[left]
            curr_char = s2[right]

            s2_count[prev_char] -= 1
            if s2_count[prev_char] == 0:
                del s2_count[prev_char]
            s2_count[curr_char] += 1
            left += 1

        return s1_count == s2_count

    # time O(n + m)
    # space O(1)

#         if len(s1) > len(s2):
#             return False

#         s1_histo = [0] * 26
#         s2_histo = [0] * 26

#         for i in range(len(s1)):
#             s1_histo[ord(s1[i]) - ord('a')] += 1
#             s2_histo[ord(s2[i]) - ord('a')] += 1

#         left = 0

#         for right in range(len(s1), len(s2)):
#             if s1_histo == s2_histo:
#                 return True

#             curr_char = s2[right]
#             prev_char = s2[left]
#             left += 1
#             s2_histo[ord(prev_char) - ord('a')] -= 1
#             s2_histo[ord(curr_char) - ord('a')] += 1


#         return s1_histo == s2_histo

# time O(n + m)
# space O(1)
