"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.



Example 1:


Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9


Constraints:

n == height.length
0 <= n <= 3 * 104
0 <= height[i] <= 105
"""


class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        lmax, rmax, left, right = 0, 0, 0, n - 1
        res = 0

        while left < right:
            if height[left] < height[right]:
                if height[left] >= lmax:
                    lmax = height[left]
                else:
                    res += lmax - height[left]

                left += 1
            else:
                if height[right] >= rmax:
                    rmax = height[right]
                else:
                    res += rmax - height[right]

                right -= 1

        return res

    # time O(n)
    # space O(1)


"""
0 1 0 2 1 0 1 3 2 1 2  1
0 1 2 3 4 5 6 7 8 9 10 11

res = 1 + 2 + 1 + 
lmax    rmax    left    right
4       10      7       10

"""
