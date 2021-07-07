"""
You are given an integer num. You can swap two digits at most once to get the maximum valued number.

Return the maximum valued number you can get.



Example 1:

Input: num = 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.
Example 2:

Input: num = 9973
Output: 9973
Explanation: No swap.


Constraints:

0 <= num <= 108
"""

from collections import defaultdict


class Solution(object):
    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """

        digs_idx = defaultdict()
        curr_num = list(str(num))

        for i, c in enumerate(str(num)):
            digs_idx[c] = i

        for i, d in enumerate(str(num)):
            for bigger_dig in range(9, int(d), -1):
                s_bigger_dig = str(bigger_dig)
                if s_bigger_dig not in digs_idx:
                    continue
                bigger_dig_idx = digs_idx[s_bigger_dig]
                if bigger_dig_idx > i:
                    curr_num[i], curr_num[bigger_dig_idx] = curr_num[bigger_dig_idx], curr_num[i]
                    return int(''.join(curr_num))

        return num

    # time O(n)
    # space O(n)

#         max_num = num
#         num = list(str(num))

#         for i in range(len(num)):
#             for j in range(i + 1, len(num)):
#                 curr_num = num[:]

#                 curr_num[i], curr_num[j] = curr_num[j], curr_num[i]
#                 curr_num = int(''.join(curr_num))
#                 if curr_num > max_num:
#                     max_num = curr_num

#         return max_num

# time O(n^2)
# space O(n)
