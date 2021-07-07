"""
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.



Example 1:

Input: num1 = "2", num2 = "3"
Output: "6"
Example 2:

Input: num1 = "123", num2 = "456"
Output: "56088"


Constraints:

1 <= num1.length, num2.length <= 200
num1 and num2 consist of digits only.
Both num1 and num2 do not contain any leading zero, except the number 0 itself.
"""


class Solution:
    def multiply(self, num1, num2):

        if num1 == '0' or num2 == '0':
            return '0'

        if len(num2) < len(num1):
            return self.multiply(num2, num1)

        num1 = [int(dig) for dig in reversed(num1)]
        num2 = [int(dig) for dig in reversed(num2)]

        res = []
        for i in range(len(num1)):
            curr_mult = [0] * i

            curr_mult += mult_digit(num1[i], num2)
            res = sum_nums(res, curr_mult)

        return ''.join([str(dig) for dig in reversed(res)])

    # time O(n1 * n2)
    # space O(n1 + n2)


def sum_nums(num1, num2):
    res = []
    carry = 0

    p1, p2 = 0, 0

    while p1 < len(num1) or p2 < len(num2):
        d1 = num1[p1] if p1 < len(num1) else 0
        d2 = num2[p2] if p2 < len(num2) else 0

        sum_ = d1 + d2 + carry
        res.append(sum_ % 10)
        carry = sum_ // 10
        p1 += 1
        p2 += 1

    if carry:
        res.append(carry)

    return res


def mult_digit(digit, num):
    res = []
    carry = 0
    for dig in num:
        mult = (digit * dig) + carry
        res.append(mult % 10)
        carry = mult // 10

    if carry:
        res.append(carry)

    return res
