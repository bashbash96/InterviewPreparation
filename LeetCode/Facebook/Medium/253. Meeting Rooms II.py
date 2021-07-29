"""
Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.



Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
Example 2:

Input: intervals = [[7,10],[2,4]]
Output: 1


Constraints:

1 <= intervals.length <= 104
0 <= starti < endi <= 106
"""


class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """

        events = []
        for time in intervals:
            events.append((time[0], 'b'))
            events.append((time[1], 'a'))

        events.sort()

        count = 0
        rooms_num = 0

        for time, event in events:
            if event == 'a':
                count -= 1
            else:
                count += 1

            rooms_num = max(rooms_num, count)

        return rooms_num

    # time O(n*log(n))
    # space O(n)
