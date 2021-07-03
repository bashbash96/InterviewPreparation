"""
Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. The result should also be sorted in ascending order.

An integer a is closer to x than an integer b if:

|a - x| < |b - x|, or
|a - x| == |b - x| and a < b


Example 1:

Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]
Example 2:

Input: arr = [1,2,3,4,5], k = 4, x = -1
Output: [1,2,3,4]


Constraints:

1 <= k <= arr.length
1 <= arr.length <= 104
arr is sorted in ascending order.
-104 <= arr[i], x <= 104
"""

from bisect import bisect_left


class Solution(object):
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """

        pivot = bisect_left(arr, x) - 1
        if pivot == -1:
            pivot = 0

        return closest_k(arr, k, x, pivot)


def closest_k(arr, k, x, pivot):
    left = pivot
    right = pivot + 1
    count = 0

    while count < k:

        if left < 0:
            right += 1
            count += 1
            continue

        if right >= len(arr) or abs(x - arr[left]) <= abs(x - arr[right]):
            left -= 1
        else:
            right += 1
        count += 1

    return arr[left + 1: right]


"""
apporach:
    1- binary search for x.
    2- expand two ways left and right searching for closest k numbers.
    time O(log(n) + k)
    space O(1)


"""
