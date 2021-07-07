"""
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.



Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Example 2:

Input: strs = [""]
Output: [[""]]
Example 3:

Input: strs = ["a"]
Output: [["a"]]


Constraints:

1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lower-case English letters.
"""

from collections import defaultdict, Counter


class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """

        word_key_map = defaultdict(list)

        for word in strs:
            key = get_key(word)
            word_key_map[key].append(word)

        res = []

        for key, val in word_key_map.items():
            res.append(val)

        return res

    # time O(n * m) m: max string length
    # space O(n * m)


def get_key(word):
    chars_count = [0] * 26

    for c in word:
        chars_count[ord(c) - ord('a')] += 1

    return tuple(chars_count)


"""
# time O(n * m log (m)) m: avg length of string
# space O(n * m)


["eat","tea","tan","ate","nat","bat"]

eat -> a1e1t1


# time O(n * (m + klog(k))) m: avg length, k: avg unique chars -> constant -> O(n * m)
# space O(n * m)


"""
