# -----------------------------------------------------------------------
"""
Problem Description

Given an array A containing N integers.

You need to find the maximum sum of triplet ( Ai + Aj + Ak ) such that 0 <= i < j < k < N and Ai < Aj < Ak.

If no such triplet exist return 0.


Problem Constraints
3 <= N <= 105.

1 <= A[i] <= 108.
Example Input
Input 1:

 A = [2, 5, 3, 1, 4, 9]

Example Output
Output 1:

 16
"""


class Solution:
    # @param A : list of integers
    # @return an integer
    def solve(self, A):

        max_sum = 0

        for i in range(1, len(A) - 1):

            prev_max, after_max = float('-inf'), float('-inf')
            for j in range(i):
                if A[j] < A[i]:
                    prev_max = max(prev_max, A[j])

            for j in range(i + 1, len(A)):
                if A[j] > A[i]:
                    after_max = max(after_max, A[j])

            if prev_max != float('-inf') and after_max != float('-inf'):
                max_sum = max(max_sum, prev_max + A[i] + after_max)

        return max_sum

    # time O(n^2)
    # space O(1)


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
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


class Solution:
    # @param intervals, a list of Intervals
    # @param new_interval, a Interval
    # @return a list of Interval

    def insert(self, intervals, new_interval):

        n = len(intervals)
        i = 0
        res = []

        # find the appropriate place
        while i < n and intervals[i].start < new_interval.start:
            res.append(Interval(intervals[i].start, intervals[i].end))
            i += 1

        # insert the new interval to the right place, merge of needed
        while res and res[-1].end > new_interval.start:
            new_interval = merge(res.pop(), new_interval)
        res.append(new_interval)

        # insert the rest values, merge if needed
        while i < n and res[-1].end > intervals[i].start:
            res.append(merge(res.pop(), intervals[i]))
            i += 1

        return res + intervals[i:]

    # time O(n)
    # space O(n)


def merge(intervalA, intervalB):
    res = [0, 0]
    res[0] = min(intervalA.start, intervalB.start)
    res[1] = max(intervalA.end, intervalB.end)

    return Interval(res[0], res[1])


# -----------------------------------------------------------------------
"""
Merge Overlapping Intervals

Given a collection of intervals, merge all overlapping intervals.

For example:

Given [1,3],[2,6],[8,10],[15,18],

return [1,6],[8,10],[15,18].

Make sure the returned intervals are sorted.
"""


class Solution:
    # @param intervals, a list of Intervals
    # @return a list of Interval
    def merge(self, intervals):
        intervals.sort(key=lambda x: (x.start, x.end))
        res = [intervals[0]]
        i = 1
        while i < len(intervals):
            curr = intervals[i]
            while res and (curr.start <= res[-1].end):
                curr = merge_intervals(res.pop(), curr)
            res.append(curr)
            i += 1
        return res

    # time O(n * log(n))
    # space O(n)


def merge_intervals(interval1, interval2):
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

        if carry:
            for i in range(1, len(A)):
                A[i] = A[i] + carry
                carry = A[i] // 10
                A[i] %= 10

        if carry:
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
    def generate(self, numRows):
        if numRows == 1:
            return [[1]]
        if numRows == 0:
            return []
        res = [[1], [1, 1]]
        while numRows > 2:
            prev = res[-1]

            res.append(get_row(prev))

            numRows -= 1

        return res

    # time O(n^2)
    # space O(n^2)


def get_row(prev_row):
    new_row = [1]

    for i in range(1, len(prev_row)):
        new_row.append(prev_row[i] + prev_row[i - 1])

    new_row.append(1)

    return new_row


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
        n = len(A)
        if n == 1:
            return 0

        l_min = [float('inf') for _ in range(n)]
        r_max = [float('-inf') for _ in range(n)]

        # build mins array
        l_min[0] = A[0]
        for i in range(1, n):
            l_min[i] = min(l_min[i - 1], A[i])

        # build maxs array
        r_max[-1] = A[-1]
        for i in range(n - 2, -1, -1):
            r_max[i] = max(A[i], r_max[i + 1])

        return get_max_diff(l_min, r_max)

    # time O(n)
    # space O(n)


def get_max_diff(l_min, r_max):
    # find the max diff
    n = len(l_min)
    min_idx, max_idx = 0, 0
    max_diff = 0
    while min_idx < n and max_idx < n:
        if l_min[min_idx] <= r_max[max_idx]:
            max_diff = max(max_diff, max_idx - min_idx)
            max_idx += 1
        else:
            min_idx += 1

    return max_diff

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

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
