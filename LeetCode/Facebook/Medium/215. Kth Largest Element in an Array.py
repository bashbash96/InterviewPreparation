"""
Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.



Example 1:

Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
Example 2:

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4


Constraints:

1 <= k <= nums.length <= 104
-104 <= nums[i] <= 104
"""

import heapq


class Solution:
    def findKthLargest(self, nums, k):

        k_largest = []
        for num in nums:
            heapq.heappush(k_largest, num)
            print(k_largest)
            if len(k_largest) > k:
                heapq.heappop(k_largest)

        return heapq.heappop(k_largest)

    # time O(n * log(k))
    # space O(k)
