"""
You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [starti, endi] and secondList[j] = [startj, endj]. Each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

A closed interval [a, b] (with a < b) denotes the set of real numbers x with a <= x <= b.

The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of [1, 3] and [2, 4] is [2, 3].



Example 1:


Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
Example 2:

Input: firstList = [[1,3],[5,9]], secondList = []
Output: []
Example 3:

Input: firstList = [], secondList = [[4,8],[10,12]]
Output: []
Example 4:

Input: firstList = [[1,7]], secondList = [[3,10]]
Output: [[3,7]]


Constraints:

0 <= firstList.length, secondList.length <= 1000
firstList.length + secondList.length >= 1
0 <= starti < endi <= 109
endi < starti+1
0 <= startj < endj <= 109
endj < startj+1
"""


class Solution(object):
    def intervalIntersection(self, firstList, secondList):
        """
        :type firstList: List[List[int]]
        :type secondList: List[List[int]]
        :rtype: List[List[int]]
        """
        n1, n2 = len(firstList), len(secondList)

        p1, p2 = 0, 0

        res = []

        while p1 < n1 and p2 < n2:

            interval1 = firstList[p1]
            interval2 = secondList[p2]

            intersection = get_intervals_intersection(interval1, interval2)

            if intersection:
                res.append(intersection)

            if interval1[1] < interval2[1]:
                p1 += 1
            else:
                p2 += 1

        return res

    # time O(min(n, m))
    # space O(min(n, m))


def get_intervals_intersection(interval1, interval2):
    x1, y1 = interval1
    x2, y2 = interval2

    if x1 > y2 or y1 < x2:
        return []

    return [max(x1, x2), min(y1, y2)]
