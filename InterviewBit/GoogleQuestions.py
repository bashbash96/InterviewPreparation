"""
Kth Row of Pascal's Triangle

Given an index k, return the kth row of the Pascal’s triangle.

Pascal’s triangle : To generate A[C] in row R, sum up A’[C] and A’[C-1] from previous row R - 1.

Example:

Input : k = 3

Return : [1,3,3,1]
"""


class Solution:
    # @param A : integer
    # @return a list of integers
    def getRow(self, k):
        if k == 0:
            return [1]
        if k == 1:
            return [1, 1]

        prev = [1, 1]
        curr = [1]
        count = 1
        while count < k:
            for i in range(len(prev) - 1):
                curr.append(prev[i] + prev[i + 1])
            curr.append(prev[-1])
            prev = [val for val in curr]
            curr = [prev[0]]

            count += 1

        return prev

        # time O(k^2)
        # space O(n) -> n is the number of elements in kth row


# -----------------------------------------------------------------------
"""
Merge Intervals

Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

Example 1:

Given intervals [1,3],[6,9] insert and merge [2,5] would result in [1,5],[6,9].

Example 2:

Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] would result in [1,2],[3,10],[12,16].

This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].

Make sure the returned intervals are also sorted.
"""


# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    # @param intervals, a list of Intervals
    # @param new_interval, a Interval
    # @return a list of Interval
    def insert(self, intervals, new_interval):
        n = len(intervals)
        i = 0
        res = []
        while i < n and new_interval.start > intervals[i].start:
            res.append(Interval(intervals[i].start, intervals[i].end))
            i += 1
        while res and res[-1].end > new_interval.start:
            new_interval = merge(res.pop(), new_interval)

        while i < n and new_interval.end > intervals[i].start:
            new_interval = merge(new_interval, intervals[i])
            i += 1

        res.append(new_interval)

        while i < n:
            res.append(intervals[i])
            i += 1

        return res

        # time O(n)
        # space O(n)


def merge(intervalA, intervalB):
    res = [0, 0]
    res[0] = min(intervalA.start, intervalB.start)

    res[1] = max(intervalA.end, intervalB.end)

    return res


# -----------------------------------------------------------------------
"""
Merge Overlapping Intervals

Given a collection of intervals, merge all overlapping intervals.

For example:

Given [1,3],[2,6],[8,10],[15,18],

return [1,6],[8,10],[15,18].

Make sure the returned intervals are sorted.

"""


# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    # @param intervals, a list of Intervals
    # @return a list of Interval
    def merge(self, intervals):
        intervals.sort(key=lambda x: (x.start, x.end))
        res = [intervals[0]]
        i = 1
        while i < len(intervals):
            curr = intervals[i]
            while res and curr.start <= res[-1].end:
                curr = merge2(res.pop(), curr)
            res.append(curr)
            i += 1
        return res

        # time O(n)
        # space O(n)


def merge2(interval1, interval2):
    res = Interval()
    res.start = min(interval1.start, interval2.start)
    res.end = max(interval1.end, interval2.end)

    return res


# -----------------------------------------------------------------------
"""
Add One To Number

Given a non-negative number represented as an array of digits,

add 1 to the number ( increment the number represented by the digits ).

The digits are stored such that the most significant digit is at the head of the list.

Example:

If the vector has [1, 2, 3]

the returned vector should be [1, 2, 4]

as 123 + 1 = 124.

 NOTE: Certain things are intentionally left unclear in this question which you should practice asking the interviewer.
For example, for this problem, following are some good questions to ask :
Q : Can the input have 0’s before the most significant digit. Or in other words, is 0 1 2 3 a valid input?
A : For the purpose of this question, YES
Q : Can the output have 0’s before the most significant digit? Or in other words, is 0 1 2 4 a valid output?
A : For the purpose of this question, NO. Even if the input has zeroes before the most significant digit.

"""


class Solution:
    # @param A : list of integers
    # @return a list of integers
    def plusOne(self, A):
        A.reverse()
        A[0] += 1
        carry = A[0] // 10
        A[0] %= 10
        for i in range(1, len(A)):
            A[i] = A[i] + carry
            carry = A[i] // 10
            A[i] %= 10

        if carry != 0:
            A.append(1)

        while A and A[-1] == 0:
            A.pop()

        A.reverse()

        return A

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
N/3 Repeat Number

You’re given a read only array of n integers. Find out if any integer occurs more than n/3 times in the array in 
linear time and constant additional space.

If so, return the integer. If not, return -1.

If there are multiple solutions, return any one.

Example :

Input : [1 2 3 1 1]
Output : 1 
1 occurs 3 times which is more than 5/3 times.
"""


class Solution:
    # @param A : tuple of integers
    # @return an integer
    def repeatedNumber(self, A):
        first, second, c1, c2 = float('inf'), float('inf'), 0, 0

        for num in A:
            if first == num:
                c1 += 1
            elif second == num:
                c2 += 1
            elif c1 == 0:
                first = num
                c1 = 1
            elif c2 == 0:
                second = num
                c2 = 1
            else:
                c1 -= 1
                c2 -= 1

        c1 = 0
        c2 = 0
        for num in A:
            if num == first:
                c1 += 1
                if c1 > len(A) // 3:
                    return first
            elif num == second:
                c2 += 1
                if c2 > len(A) // 3:
                    return second

        return -1

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
Max Distance

Given an array A of integers, find the maximum of j - i subjected to the constraint of A[i] <= A[j].

If there is no solution possible, return -1.

Example :

A : [3 5 4 2]

Output : 2 
for the pair (3, 4)
"""


class Solution:
    # @param A : tuple of integers
    # @return an integer
    def maximumGap(self, A):
        if len(A) == 1:
            return 0
        n = len(A)
        lMin = [0 for i in A]
        rMax = [0 for i in A]
        lMin[0] = A[0]
        for i in range(1, n):
            lMin[i] = min(A[i], lMin[i - 1])

        rMax[-1] = A[-1]
        for i in range(n - 2, -1, -1):
            rMax[i] = max(rMax[i + 1], A[i])

        i, j = 0, 0
        maxGap = 0
        while j < n and i < n:
            if rMax[j] >= lMin[i]:
                maxGap = max(maxGap, j - i)
                j = j + 1
            else:
                i = i + 1

        return maxGap


# -----------------------------------------------------------------------
"""
Pascal Triangle

Given numRows, generate the first numRows of Pascal’s triangle.

Pascal’s triangle : To generate A[C] in row R, sum up A’[C] and A’[C-1] from previous row R - 1.

Example:

Given numRows = 5,

Return

[
     [1],
     [1,1],
     [1,2,1],
     [1,3,3,1],
     [1,4,6,4,1]
]
"""


class Solution:
    # @param A : integer
    # @return a list of list of integers
    def solve(self, A):
        if A <= 0:
            return []
        res = [[1]]
        for i in range(A - 1):
            curr = [res[i][0]]
            for j in range(1, len(res[i])):
                curr.append(res[i][j] + res[i][j - 1])
            curr.append(res[i][-1])
            res.append(curr)

        return res

        # time O(n)
        # space O(n^2)


# -----------------------------------------------------------------------
"""
Max Non Negative SubArray

Given an array of integers, A of length N, find out the maximum sum sub-array of non negative numbers from A.

The sub-array should be contiguous i.e., a sub-array created by choosing the second and fourth element and skipping the 
third element is invalid.

Maximum sub-array is defined in terms of the sum of the elements in the sub-array.

Find and return the required subarray.

NOTE:

    1. If there is a tie, then compare with segment's length and return segment which has maximum length.
    2. If there is still a tie, then return the segment with minimum starting index.


Input Format:

The first and the only argument of input contains an integer array A, of length N.
Output Format:

Return an array of integers, that is a subarray of A that satisfies the given conditions.
Constraints:

1 <= N <= 1e5
1 <= A[i] <= 1e5
Examples:

Input 1:
    A = [1, 2, 5, -7, 2, 3]

Output 1:
    [1, 2, 5]

Explanation 1:
    The two sub-arrays are [1, 2, 5] [2, 3].
    The answer is [1, 2, 5] as its sum is larger than [2, 3].

Input 2:
    A = [10, -1, 2, 3, -4, 100]
    
Output 2:
    [100]

Explanation 2:
    The three sub-arrays are [10], [2, 3], [100].
    The answer is [100] as its sum is larger than the other two.
"""


class Solution:
    # @param A : list of integers
    # @return a list of integers
    def maxset(self, A):
        currList, currSum = [], 0
        resList, resSum = [], 0
        for num in A:
            if num < 0:
                currList = []
                currSum = 0
            else:
                currList.append(num)
                currSum += num
                if currSum > resSum:
                    resList = currList
                    resSum = currSum
                elif currSum == resSum:
                    if len(currList) > len(resList):
                        resList = currList
                        resSum = currSum
        return resList

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
Wave Array

Given an array of integers, sort the array into a wave like array and return it,
In other words, arrange the elements into a sequence such that a1 >= a2 <= a3 >= a4 <= a5.....

Example

Given [1, 2, 3, 4]

One possible answer : [2, 1, 4, 3]
Another possible answer : [4, 1, 3, 2]
 NOTE : If there are multiple answers possible, return the one thats lexicographically smallest.
So, in example case, you will return [2, 1, 4, 3] 
"""


class Solution:
    # @param A : list of integers
    # @return a list of integers
    def wave(self, A):
        if len(A) < 2:
            return A

        A.sort()
        for i in range(0, len(A) - 1, 2):
            A[i], A[i + 1] = A[i + 1], A[i]
        return A

        # time O(n * log(n))
        # space O(1)


# -----------------------------------------------------------------------
"""
Sum of pairwise Hamming Distance

Hamming distance between two non-negative integers is defined as the number of positions at which the corresponding bits are different.

For example,

HammingDistance(2, 7) = 2, as only the first and the third bit differs in the binary representation of 2 (010) and 7 (111).

Given an array of N non-negative integers, find the sum of hamming distances of all pairs of integers in the array.
Return the answer modulo 1000000007.

Example

Let f(x, y) be the hamming distance defined above.

A=[2, 4, 6]

We return,
f(2, 2) + f(2, 4) + f(2, 6) + 
f(4, 2) + f(4, 4) + f(4, 6) +
f(6, 2) + f(6, 4) + f(6, 6) = 

0 + 2 + 1
2 + 0 + 1
1 + 1 + 0 = 8
"""

numOvBits = 32
MOD = 1000000007


class Solution:
    # @param A : tuple of integers
    # @return an integer
    def hammingDistance(self, A):
        A = list(A)
        n = len(A)
        res = 0

        for i in range(len(A)):
            A[i] = getBits(A[i])

        for j in range(numOvBits):
            counter = 0
            for i in range(n):
                if A[i][j] == 0:
                    counter += 1
            res += ((2 * counter * (n - counter)) % MOD)
            res %= MOD
        return res

        # time O(n)
        # space O(1)


def getBits(num):
    res = []
    while num > 0:
        res.append(num & 1)
        num >>= 1
    if len(res) < numOvBits:
        res.extend([0] * (numOvBits - len(res)))
    res.reverse()
    return res


# -----------------------------------------------------------------------
"""
Rotate Matrix
You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

You need to do this in place.

Note that if you end up using an additional array, you will only receive partial score.

Example:

If the array is

[
    [1, 2],
    [3, 4]
]
Then the rotated array becomes:

[
    [3, 1],
    [4, 2]
]
"""


class Solution:
    # @param A : list of list of integers
    # @return the same list modified
    def rotate(self, A):

        for row in range(len(A) // 2):
            for col in range(row, len(A) - row - 1):
                rotate(A, row, col)

        return A

    # time O(n^2)
    # space O(1)


def rotate(mat, row, col):
    rows = len(mat)
    topLeft = mat[row][col]
    mat[row][col] = mat[rows - col - 1][row]
    mat[rows - col - 1][row] = mat[rows - row - 1][rows - col - 1]
    mat[rows - row - 1][rows - col - 1] = mat[col][rows - row - 1]
    mat[col][rows - row - 1] = topLeft

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
