"""
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.


Example:

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True


Constraints:

1 <= word.length <= 500
word in addWord consists lower-case English letters.
word in search consist of  '.' or lower-case English letters.
At most 50000 calls will be made to addWord and search.
"""


class Trie:
    def __init__(self):
        self.root = {}

    def add(self, word):
        curr = self.root
        for c in word:
            if c not in curr:
                curr[c] = {}

            curr = curr[c]

        curr['*'] = '*'

    def search(self, word):

        curr = self.root
        return self._search(word, 0, curr)

    def _search(self, word, idx, curr):

        if curr is None:
            return False

        if idx == len(word):
            return '*' in curr

        curr_char = word[idx]

        if curr_char != '.' and curr_char not in curr:
            return False

        if curr_char == '.':
            for c in curr:
                if c == '*':
                    continue
                if self._search(word, idx + 1, curr[c]):
                    return True

            return False

        return self._search(word, idx + 1, curr[curr_char])


class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """

        self.trie = Trie()

    def addWord(self, word):
        """
        :type word: str
        :rtype: None
        """
        self.trie.add(word)

    # time O(n)
    # space O(n)

    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """

        return self.trie.search(word)

    # time O(~n)
    # space O(n)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
