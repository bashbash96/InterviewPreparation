"""
Convert a non-negative integer num to its English words representation.



Example 1:

Input: num = 123
Output: "One Hundred Twenty Three"
Example 2:

Input: num = 12345
Output: "Twelve Thousand Three Hundred Forty Five"
Example 3:

Input: num = 1234567
Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
Example 4:

Input: num = 1234567891
Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"


Constraints:

0 <= num <= 231 - 1
Accepted
247,524
Submissions
863,338
"""

numbers_to_words = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven',
                    'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']

tens = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']

bigs = ['Hundred', 'Thousand', 'Million', 'Billion', 'Trillion']


class Solution(object):
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """

        num = str(num)[::-1]

        res = []
        bigs_idx = 0
        while num and bigs_idx < len(bigs):

            if bigs_idx == 0:
                words = get_words(num[:3][::-1])
            else:
                curr = get_words(num[:3][::-1])
                words = "{} {}".format(curr, bigs[bigs_idx]) if curr else ''

            if words != '':
                res.append(words)

            num = num[3:]
            bigs_idx += 1

        return ' '.join(res[::-1])

    # time O(n)
    # space O(n)


def get_words(num):
    if len(num) < 3:
        return two_digs(num)

    if num[0] == '0':
        return two_digs(num[1:])

    hundreds = numbers_to_words[int(num[0])] + ' ' + bigs[0]
    twos = two_digs(num[1:])
    if twos == '':
        return hundreds

    return hundreds + ' ' + two_digs(num[1:])


def two_digs(num):
    if int(num) == 0 and len(num) > 1:
        return ''

    if int(num) < 20:
        return numbers_to_words[int(num)]

    res = []
    res.append(tens[int(num[0])])

    if num[1] != '0':
        res.append(numbers_to_words[int(num[1])])

    return ' '.join(res)
