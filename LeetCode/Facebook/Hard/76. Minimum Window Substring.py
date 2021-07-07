"""
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

A substring is a contiguous sequence of characters within the string.



Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.


Constraints:

m == s.length
n == t.length
1 <= m, n <= 105
s and t consist of uppercase and lowercase English letters.


Follow up: Could you find an algorithm that runs in O(m + n) time?
"""

from collections import Counter, defaultdict


class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """

        n = len(s)

        if len(s) < len(t):
            return ''

        t_counts = Counter(t)
        s_counts = defaultdict(int)
        completed_chars = 0

        left = 0
        start_idx, length = 0, float('inf')

        for right in range(n):
            curr_char = s[right]

            s_counts[curr_char] += 1

            if s_counts[curr_char] == t_counts[curr_char]:
                completed_chars += 1

            # found all chars with all frequencies
            if completed_chars == len(t_counts):

                while left < right and (s[left] not in t_counts or s_counts[s[left]] > t_counts[s[left]]):
                    s_counts[s[left]] -= 1
                    left += 1

                curr_length = right - left + 1
                if curr_length < length:
                    start_idx = left
                    length = curr_length

                s_counts[s[left]] -= 1
                completed_chars -= 1
                left += 1

        return s[start_idx: start_idx + length] if length != float('inf') else ''

    # time O(s)
    # space O(s)


"""

"ADOBECODEBANC"
          *  *
"ABC"

t_counts    completed_chars        left    right       s_counts
{A: 1,             3                8        11        {A: 1, D: 0, O: 0, B: 1, E: 0, C: 1, N: 1
B: 1,                           
C: 1}


start_idx       length
    8              4

"""
