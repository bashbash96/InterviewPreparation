"""
Given a string s, find the length of the longest substring without repeating characters.


Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
Example 4:

Input: s = ""
Output: 0


Constraints:

0 <= s.length <= 5 * 104
s consists of English letters, digits, symbols and spaces.
"""


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """

        seen_chars = set()
        longest_sub = 0
        left = 0

        for right, curr_char in enumerate(s):

            if curr_char in seen_chars:
                while left < right and curr_char in seen_chars:
                    seen_chars.discard(s[left])
                    left += 1

            seen_chars.add(curr_char)

            curr_length = right - left + 1

            longest_sub = max(longest_sub, curr_length)

        return longest_sub

    # time O(n)
    # space O(n)
