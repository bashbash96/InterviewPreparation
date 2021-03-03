# -----------------------------------------------------------------------
"""
Longest Common Prefix

Given the array of strings A,
you need to find the longest string S which is the prefix of ALL the strings in the array.

Longest common prefix for a pair of strings S1 and S2 is the longest string S which is the prefix of both S1
and S2.

For Example, longest common prefix of "abcdefgh" and "abcefgh" is "abc".



Input Format

The only argument given is an array of strings A.
Output Format

Return longest common prefix of all strings in A.
For Example

Input 1:
    A = ["abcdefgh", "aefghijk", "abcefgh"]
Output 1:
    "a"
    Explanation 1:
        Longest common prefix of all the strings is "a".

Input 2:
    A = ["abab", "ab", "abcd"];
Output 2:
    "ab"
    Explanation 2:
        Longest common prefix of all the strings is "ab".
"""


class Solution:
    # @param A : list of strings
    # @return a strings
    def longestCommonPrefix(self, A):

        if len(A) < 1:
            return ''

        if len(A) == 1:
            return A[0]

        lcp = get_lcp(A[0], A[1])

        for i in range(2, len(A)):
            lcp = get_lcp(lcp, A[i])

            if lcp == '':
                break

        return lcp

    # time O(n * k) k: |lcp|
    # space O(k)


def get_lcp(str1, str2):
    p = 0

    while p < len(str1) and p < len(str2) and str1[p] == str2[p]:
        p += 1

    return str1[:p]


# -----------------------------------------------------------------------
"""
Multiply Strings

Given two numbers represented as strings, return multiplication of the numbers as a string.

 Note: The numbers can be arbitrarily large and are non-negative.
Note2: Your answer should not have leading zeroes. For example, 00 is not a valid answer. 
For example,
given strings "12", "10", your answer should be “120”.

NOTE : DO NOT USE BIG INTEGER LIBRARIES ( WHICH ARE AVAILABLE IN JAVA / PYTHON ).
We will retroactively disqualify such submissions and the submissions will incur penalties.
"""


class Solution:
    # @param A : string
    # @param B : string
    # @return a strings
    def multiply(self, A, B):

        if A == '0' or B == '0':
            return '0'

        if len(B) < len(A):
            return self.multiply(B, A)

        num1 = list(A)[::-1]
        num2 = list(B)[::-1]

        res = []
        for i, digit in enumerate(num1):
            curr_res = [0] * i  # add appropriate zeros
            curr_res += mult_digit(num2, digit)
            res = add_nums(res, curr_res)

        while res and res[-1] == '0':
            res.pop()

        res.reverse()

        return ''.join(res)

    # time O(a * b)
    # space O(a * b)


def mult_digit(num, digit):
    if not num:
        return []

    c_num = [int(val) for val in num]
    digit = int(digit)

    res = []
    mult, carry = 0, 0

    for dig in c_num:
        mult = dig * digit + carry
        carry = mult // 10
        mult %= 10
        res.append(str(mult))

    if carry:
        res.append(str(carry))

    return res


def add_nums(num1, num2):
    c_num1 = [int(val) for val in num1]
    c_num2 = [int(val) for val in num2]

    p1, p2 = 0, 0
    curr_sum, carry = 0, 0

    res = []
    while p1 < len(c_num1) or p2 < len(c_num2):
        val1 = c_num1[p1] if p1 < len(c_num1) else 0
        val2 = c_num2[p2] if p2 < len(c_num2) else 0

        curr_sum = val1 + val2 + carry
        carry = curr_sum // 10
        curr_sum %= 10

        res.append(str(curr_sum))

        p1 += 1
        p2 += 1

    if carry:
        res.append(str(carry))

    return res


# -----------------------------------------------------------------------
"""
Justified Text

Given an array of words and a length L, format the text such that each line has exactly L characters and is fully (left and right) justified.
You should pack your words in a greedy approach; that is, pack as many words as you can in each line.

Pad extra spaces ‘ ‘ when necessary so that each line has exactly L characters.
Extra spaces between words should be distributed as evenly as possible.
If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
For the last line of text, it should be left justified and no extra space is inserted between words.

Your program should return a list of strings, where each string represents a single line.

Example:

words: ["This", "is", "an", "example", "of", "text", "justification."]

L: 16.

Return the formatted lines as:

[
   "This    is    an",
   "example  of text",
   "justification.  "
]
 Note: Each word is guaranteed not to exceed L in length. 
"""


class Solution:
    # @param A : list of strings
    # @param B : integer
    # @return a list of strings
    def fullJustify(self, A, B):
        res = []
        texts = A
        max_len = B

        curr_len = 0
        curr_texts = []
        for word in texts:
            if curr_len + len(word) <= max_len:
                curr_texts.append(word)
                curr_len += len(word) + 1  # extra space for each word
            else:
                curr_len = len(word) + 1
                add_to_res(res, curr_texts, max_len)
                curr_texts = [word]

        add_last_line_to_res(res, curr_texts, max_len)

        return res

    # time O(n * k)
    # space O(n * k)


def add_last_line_to_res(res, curr_texts, max_len):
    if not curr_texts:
        return

    curr = []
    total_len = sum([len(val) for val in curr_texts])

    total_spaces = max_len - (total_len + len(curr_texts) - 1)
    for word in curr_texts:
        curr.append(word)
        curr.append(' ')
    curr.pop()
    curr.extend([' '] * total_spaces)
    res.append(''.join(curr))


def add_to_res(res, curr_texts, max_len):
    if not curr_texts:
        return

    curr = []
    total_len = sum([len(val) for val in curr_texts])

    if len(curr_texts) == 1:
        total_spaces = max_len - total_len
        curr.append(curr_texts[0])
        curr.extend([' '] * total_spaces)
        res.append(''.join(curr))
        return

    total_spaces = (max_len - total_len)
    word_spaces = total_spaces // (len(curr_texts) - 1)

    extra_spaces = total_spaces - word_spaces * (len(curr_texts) - 1)

    for i in range(len(curr_texts) - 1):
        curr.append(curr_texts[i])
        if extra_spaces > 0:
            curr.extend([' '] * (word_spaces + 1))
            extra_spaces -= 1
        else:
            curr.extend([' '] * word_spaces)

    curr.append(curr_texts[-1])

    res.append(''.join(curr))


# -----------------------------------------------------------------------
"""
Minimum Parantheses!

Given a string A of parantheses ‘(‘ or ‘)’.

The task is to find minimum number of parentheses ‘(‘ or ‘)’ (at any positions) we must add to make the resulting parentheses string valid.

An string is valid if:

Open brackets must be closed by the corresponding closing bracket.
Open brackets must be closed in the correct order.


Problem Constraints
1 <= |A| <= 105

A[i] = '(' or A[i] = ')'



Input Format
First and only argument is an string A.



Output Format
Return a single integer denoting the minimumnumber of parentheses ‘(‘ or ‘)’ (at any positions) we must add in A to make the resulting parentheses string valid.



Example Input
Input 1:

 A = "())"
Input 2:

 A = "((("


Example Output
Output 1:

 1
Output 2:

 3
"""


class Solution:
    # @param A : string
    # @return an integer
    def solve(self, A):

        counter = 0
        res = 0
        for char in A:
            if char == '(':
                counter += 1
            else:
                counter -= 1
                if counter < 0:
                    res += 1
                    counter = 0

        return res + counter

    # time O(n)
    # space O(1)

# -----------------------------------------------------------------------
