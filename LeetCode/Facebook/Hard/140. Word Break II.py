"""
Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

Note that the same word in the dictionary may be reused multiple times in the segmentation.



Example 1:

Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
Example 2:

Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
Explanation: Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: []


Constraints:

1 <= s.length <= 20
1 <= wordDict.length <= 1000
1 <= wordDict[i].length <= 10
s and wordDict[i] consist of only lowercase English letters.
All the strings of wordDict are unique.
"""


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """

        return generate_valid_sentences(s, set(wordDict), 0, {})

    # time O(n^n * n)
    # space O(n^n * n)


def generate_valid_sentences(string, words, idx, memo):
    if idx in memo:
        return memo[idx]

    if idx == len(string):
        return []

    memo[idx] = []
    for end in range(idx + 1, len(string) + 1):

        curr = string[idx:end]
        if curr in words:
            rest = generate_valid_sentences(string, words, end, memo)

            if rest:
                for sentence in rest:
                    memo[idx].append(curr + ' ' + sentence)
            elif end == len(string):
                memo[idx].append(curr)

    return memo[idx]
