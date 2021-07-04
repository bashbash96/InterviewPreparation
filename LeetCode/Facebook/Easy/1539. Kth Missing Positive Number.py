"""
Given an array arr of positive integers sorted in a strictly increasing order, and an integer k.

Find the kth positive integer that is missing from this array.



Example 1:

Input: arr = [2,3,4,7,11], k = 5
Output: 9
Explanation: The missing positive integers are [1,5,6,8,9,10,12,13,...]. The 5th missing positive integer is 9.
Example 2:

Input: arr = [1,2,3,4], k = 2
Output: 6
Explanation: The missing positive integers are [5,6,7,...]. The 2nd missing positive integer is 6.


Constraints:

1 <= arr.length <= 1000
1 <= arr[i] <= 1000
1 <= k <= 1000
arr[i] < arr[j] for 1 <= i < j <= arr.length
"""


class Solution(object):
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """

        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) >> 1
            missing_elements = arr[mid] - mid - 1

            if missing_elements < k:
                left = mid + 1
            else:
                right = mid - 1

        return left + k

    # time O(log(n))
    # space O(1)


#         missing_order = 0
#         curr_num = 1

#         i = 0
#         while i < len(arr):
#             if arr[i] < curr_num:
#                 i += 1
#             elif arr[i] == curr_num:
#                 i += 1
#                 curr_num += 1
#             else:
#                 missing_order += 1
#                 if missing_order == k:
#                     return curr_num
#                 curr_num += 1

#         while missing_order < k:
#             missing_order += 1
#             curr_num += 1

#         return curr_num - 1

#     # time O(n)
#     # space O(1)


"""
[1,4,6,7,8,20]
 1 2 3 4 5 6 7 8 9 10 ....

 k = 4
mising: [2,3,5,9] => return 9

dry run:

i   arr[i]      curr_num    missing_order
0   1               1           0
1   4               2           0
1   4               3           1
1   4               4           2
2   6               5           2
2   6               6           3
3   7               7           3
4   8               8           3
5   20              9           3
                                4 => return 9
"""
