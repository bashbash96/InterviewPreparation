# -----------------------------------------------------------------------
"""
Gas Station

Given two integer arrays A and B of size N.
There are N gas stations along a circular route, where the amount of gas at station i is A[i].

You have a car with an unlimited gas tank and it costs B[i] of gas to travel from station i
to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.

Return the minimum starting gas station’s index if you can travel around the circuit once, otherwise return -1.

You can only travel in one direction. i to i+1, i+2, … n-1, 0, 1, 2.. Completing the circuit means starting at i and
ending up at i again.



Input Format

The first argument given is the integer array A.
The second argument given is the integer array B.
Output Format

Return the minimum starting gas station's index if you can travel around the circuit once, otherwise return -1.
For Example

Input 1:
    A =  [1, 2]
    B =  [2, 1]
Output 1:
    1

"""


class Solution:
    # @param A : tuple of integers
    # @param B : tuple of integers
    # @return an integer
    def canCompleteCircuit(self, A, B):
        n = len(A)
        if n == 0:
            return -1
        if n == 1:
            return 0 if A[0] - B[0] >= 0 else -1

        gas = A
        cost = B
        start, end = 0, 1
        curr_tank = gas[start] - cost[start]
        while start != end:

            while curr_tank < 0 and start != end:
                curr_tank -= (gas[start] - cost[start])
                start = (start + 1) % n

                if start == 0:
                    return -1

            curr_tank += gas[end] - cost[end]
            end = (end + 1) % n

        if curr_tank < 0:
            return -1

        return start

    # time O(n)
    # space O(1)

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
