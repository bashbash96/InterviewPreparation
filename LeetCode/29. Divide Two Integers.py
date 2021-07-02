"""
Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero, which means losing its fractional part. For example, truncate(8.345) = 8 and truncate(-2.7335) = -2.

Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−231, 231 − 1]. For this problem, assume that your function returns 231 − 1 when the division result overflows.



Example 1:

Input: dividend = 10, divisor = 3
Output: 3
Explanation: 10/3 = truncate(3.33333..) = 3.
Example 2:

Input: dividend = 7, divisor = -3
Output: -2
Explanation: 7/-3 = truncate(-2.33333..) = -2.
Example 3:

Input: dividend = 0, divisor = 1
Output: 0
Example 4:

Input: dividend = 1, divisor = 1
Output: 1


Constraints:

-231 <= dividend, divisor <= 231 - 1
divisor != 0
"""

MAX = pow(2, 31)


class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """

        if divisor == 0:
            raise ValueError("Invalid input")

        if divisor == -1 or divisor == 1:
            return check_if_flows(divisor * dividend)

        final_sign = get_sign(dividend, divisor)

        dividend, divisor = make_positive(dividend, divisor)

        res = 0

        while dividend >= divisor:

            powers = 1
            val = divisor

            while val + val < dividend:
                val += val
                powers += powers

            res += powers
            dividend -= val

        return check_if_flows(res) * final_sign

    # time O(log(n))
    # space O(1)


def get_sign(dividend, divisor):
    if dividend < 0 and divisor < 0:
        return 1

    if dividend < 0 or divisor < 0:
        return -1

    return 1


def make_positive(dividend, divisor):
    if dividend < 0:
        dividend *= -1

    if divisor < 0:
        divisor *= -1

    return dividend, divisor


def check_if_flows(res):
    if res < -MAX:
        return -MAX

    if res > (MAX - 1):
        return MAX - 1

    return res


"""

dividend  10
divisor 3
=> 3


approach 1:

    1. while div larger than d:
        1.1. add one to count.
        1.2 reduce div by d
    2. return count

# time O(max(dividend, divisor))
# space O(1)




"""
