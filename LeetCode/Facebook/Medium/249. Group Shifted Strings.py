"""
We can shift a string by shifting each of its letters to its successive letter.

For example, "abc" can be shifted to be "bcd".
We can keep shifting the string to form a sequence.

For example, we can keep shifting "abc" to form the sequence: "abc" -> "bcd" -> ... -> "xyz".
Given an array of strings strings, group all strings[i] that belong to the same shifting sequence. You may return the answer in any order.



Example 1:

Input: strings = ["abc","bcd","acef","xyz","az","ba","a","z"]
Output: [["acef"],["a","z"],["abc","bcd","xyz"],["az","ba"]]
Example 2:

Input: strings = ["a"]
Output: [["a"]]


Constraints:

1 <= strings.length <= 200
1 <= strings[i].length <= 50
strings[i] consists of lowercase English letters.
"""
from collections import defaultdict
from bisect import insort


class Solution(object):
    def groupStrings(self, strings):
        """
        :type strings: List[str]
        :rtype: List[List[str]]
        """

        sequences = get_sequences_by_length(strings)
        res = []

        for sequence in sequences:
            res.extend(get_sequences_by_shifting(sequence))

        return res

    # time O(n * l) -> n: num of words, l avg length
    # space O(n * m)


def get_sequences_by_length(strings):
    sequences = defaultdict(list)

    for word in strings:
        insort(sequences[len(word)], word)

    return sequences.values()


def get_sequences_by_shifting(sequence):
    res = defaultdict(list)

    for word in sequence:
        shift = get_shift(word)
        res[shift].append(word)

    return res.values()


def get_shift(word):
    shift = []

    for i in range(len(word) - 1):
        diff = (ord(word[i + 1]) - ord(word[i]))
        if diff < 0:
            diff += 26

        shift.append(str(diff))

    return ''.join(shift)
