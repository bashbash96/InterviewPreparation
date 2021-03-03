# -----------------------------------------------------------------------
"""

Evaluate Expression

Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are +, -, *, /. Each operand may be an integer or another expression.



Input Format

The only argument given is character array A.
Output Format

Return the value of arithmetic expression formed using reverse Polish Notation.
For Example

Input 1:
    A =   ["2", "1", "+", "3", "*"]
Output 1:
    9
Explaination 1:
    starting from backside:
    *: ( )*( )
    3: ()*(3)
    +: ( () + () )*(3)
    1: ( () + (1) )*(3)
    2: ( (2) + (1) )*(3)
    ((2)+(1))*(3) = 9

Input 2:
    A = ["4", "13", "5", "/", "+"]
Output 2:
    6
Explaination 2:
    +: ()+()
    /: ()+(() / ())
    5: ()+(() / (5))
    1: ()+((13) / (5))
    4: (4)+((13) / (5))
    (4)+((13) / (5)) = 6
"""


class Solution:
    # @param A : list of strings
    # @return an integer
    def evalRPN(self, A):
        nums = []
        ops = ['+', '-', '*', '/']

        for char in A:
            if char in ops:
                if len(nums) < 2:
                    raise ('Invalid input')
                nums.append(int(calculate_operation(nums.pop(), nums.pop(), char)))
            else:
                nums.append(int(char))

        return nums.pop()

    # time O(n)
    # space O(n)


def calculate_operation(val1, val2, operation):
    if operation == '+':
        return val1 + val2
    elif operation == '-':
        return val2 - val1
    elif operation == '*':
        return val1 * val2
    else:
        return val2 / val1


# -----------------------------------------------------------------------
"""
Sliding Window Maximum

Given an array of integers A. There is a sliding window of size B which
is moving from the very left of the array to the very right.
You can only see the w numbers in the window. Each time the sliding window moves
rightwards by one position. You have to find the maximum for each window.
The following example will give you more clarity.

The array A is [1 3 -1 -3 5 3 6 7], and B is 3.

Window position	Max
———————————-	————————-
[1 3 -1] -3 5 3 6 7	3
1 [3 -1 -3] 5 3 6 7	3
1 3 [-1 -3 5] 3 6 7	5
1 3 -1 [-3 5 3] 6 7	5
1 3 -1 -3 [5 3 6] 7	6
1 3 -1 -3 5 [3 6 7]	7
Return an array C, where C[i] is the maximum value of from A[i] to A[i+B-1].

Note: If B > length of the array, return 1 element with the max of the array.



Input Format

The first argument given is the integer array A.
The second argument given is the integer B.
Output Format

Return an array C, where C[i] is the maximum value of from A[i] to A[i+B-1]
"""

from collections import deque


class Solution:
    # @param A : tuple of integers
    # @param B : integer
    # @return a list of integers
    def slidingMaximum(self, A, B):
        res = []
        nums = A
        k = B
        if k == 1:
            return nums

        q = deque()
        for i in range(k):
            add_to_queue(q, nums, i)

        left = 0
        for i in range(k, len(nums)):

            res.append(nums[q[0]])

            while q and q[0] <= left:
                q.popleft()

            left += 1

            add_to_queue(q, nums, i)

        res.append(nums[q[0]])

        return res

    # time O(n)
    # space O(k)


def add_to_queue(q, nums, i):
    while q and nums[i] >= nums[q[-1]]:
        q.pop()
    q.append(i)


# -----------------------------------------------------------------------
"""
Balanced Parantheses!

Given a string A consisting only of '(' and ')'.

You need to find whether parantheses in A is balanced or not ,if it is balanced then return 1 else return 0.



Problem Constraints
1 <= |A| <= 105



Input Format
First argument is an string A.



Output Format
Return 1 if parantheses in string are balanced else return 0.



Example Input
Input 1:

 A = "(()())"
Input 2:

 A = "(()"


Example Output
Output 1:

 1
Output 2:

 0
"""


class Solution:
    # @param A : string
    # @return an integer
    def solve(self, A):
        counter = 0

        for char in A:
            if char == '(':
                counter += 1
            else:
                counter -= 1

            if counter < 0:
                return 0

        return 1 if counter == 0 else 0

    # time O(n)
    # space O(1)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
