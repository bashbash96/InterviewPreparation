import collections
import sys

# -----------------------------------------------------------------------
"""
525. Contiguous Array

Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.

Example 1:
Input: [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with equal number of 0 and 1.
"""


class Solution:
    def findMaxLength(self, nums):
        map = {}
        count, maxLen = 0, 0
        map[0] = -1
        for i in range(len(nums)):
            count += (1 if nums[i] == 1 else -1)
            if count in map:
                maxLen = max(maxLen, i - map[count])
            else:
                map[count] = i

        return maxLen

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
904. Fruit Into Baskets

In a row of trees, the i-th tree produces fruit with type tree[i].

You start at any tree of your choice, then repeatedly perform the following steps:

Add one piece of fruit from this tree to your baskets.  If you cannot, stop.
Move to the next tree to the right of the current tree.  If there is no tree to the right, stop.
Note that you do not have any choice after the initial choice of starting tree: you must perform step 1, 
then step 2, then back to step 1, then step 2, and so on until you stop.

You have two baskets, and each basket can carry any quantity of fruit, but you want each basket to 
only carry one type of fruit each.

What is the total amount of fruit you can collect with this procedure?

 

Example 1:

Input: [1,2,1]
Output: 3
Explanation: We can collect [1,2,1].
"""


class Solution:
    def totalFruit(self, tree):
        idx, maxLen, map = 0, 0, collections.defaultdict(int)

        for i in range(len(tree)):
            map[tree[i]] += 1
            while len(map) >= 3:
                map[tree[idx]] -= 1
                if map[tree[idx]] == 0:
                    del map[tree[idx]]
                idx += 1
            maxLen = max(maxLen, i - idx + 1)

        return maxLen

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
Given an array of roses. roses[i] means rose i will bloom on day roses[i]. Also given an int k, which is the minimum 
number of adjacent bloom roses required for a bouquet, and an int n, which is the number of bouquets we need. 
Return the earliest day that we can get n bouquets of roses.

Example:
Input: roses = [1, 2, 4, 9, 3, 4, 1], k = 2, n = 2
Output: 4
Explanation:
day 1: [b, n, n, n, n, n, b]
The first and the last rose bloom.

day 2: [b, b, n, n, n, n, b]
The second rose blooms. Here the first two bloom roses make a bouquet.

day 3: [b, b, n, n, b, n, b]

day 4: [b, b, b, n, b, b, b]
Here the last three bloom roses make a bouquet, meeting the required n = 2 bouquets of bloom roses. So return day 4.
"""


def earliestDay(roses, k, n):
    start, end = 1, max(roses)
    day = end

    while start <= end:
        mid = (start + end) // 2
        if isEnoughBouquets(roses, mid, k, n):
            day = min(day, mid)
            end = mid - 1
        else:
            start = mid + 1

    return day

    # time O(n * log(n))
    # space O(1)


def isEnoughBouquets(roses, day, k, n):
    currCount, res = 0, 0

    for d in roses:
        if d <= day:
            currCount += 1
            if currCount == k:
                res += 1
                currCount = 0
        else:
            currCount = 0
        if res >= n:
            return True

    return False


# -----------------------------------------------------------------------
"""
Given an int array nums of length n. Split it into strictly decreasing subsequences. 
Output the min number of subsequences you can get by splitting.

Example 1:

Input: [5, 2, 4, 3, 1, 6]
Output: 3
Explanation:
You can split this array into: [5, 2, 1], [4, 3], [6]. And there are 3 subsequences you get.
Or you can split it into [5, 4, 3], [2, 1], [6]. Also 3 subsequences.
But [5, 4, 3, 2, 1], [6] is not legal because [5, 4, 3, 2, 1] is not a subsuquence of the original array.

Example 2:

Input: [2, 9, 12, 13, 4, 7, 6, 5, 10]
Output: 4
Explanation: [2], [9, 4], [12, 10], [13, 7, 6, 5]

Example 3:

Input: [1, 1, 1]
Output: 3
Explanation: Because of the strictly descending order you have to split it into 3 subsequences: [1], [1], [1]
"""


def minSub(arr):
    res = 0

    for i in range(len(arr)):
        if arr[i] == sys.maxsize:
            continue
        currNum = arr[i]
        arr[i] = sys.maxsize
        for j in range(i + 1, len(arr)):
            if arr[j] != sys.maxsize and arr[j] < currNum:
                currNum = arr[j]
                arr[j] = sys.maxsize
        res += 1

    return res

    # time O(n^2)
    # space O(1)


# -----------------------------------------------------------------------
"""
300. Longest Increasing Subsequence

Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:

Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
Note:

There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.
Follow up: Could you improve it to O(n log n) time complexity?
"""


class Solution:
    def lengthOfLIS(self, nums):

        memo = [1] * len(nums)
        LIS = 0
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    memo[i] = max(memo[i], memo[j] + 1)
            LIS = max(LIS, memo[i])

        return LIS

        # time O(n^2)
        # space O(n)


# -----------------------------------------------------------------------
"""
238. Product of Array Except Self

Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of 
all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Constraint: It's guaranteed that the product of the elements of any prefix or suffix of the array 
(including the whole array) fits in a 32 bit integer.

Note: Please solve it without division and in O(n).

Follow up:
Could you solve it with constant space complexity? 
(The output array does not count as extra space for the purpose of space complexity analysis.)
"""


class Solution:
    def productExceptSelf(self, nums):
        n = len(nums)
        left, right = [0] * n, [0] * n

        currProd = 1
        for i in range(n):
            left[i] = currProd
            currProd *= nums[i]

        currProd = 1
        for i in range(n - 1, -1, -1):
            right[i] = currProd
            currProd *= nums[i]

        for i in range(n):
            nums[i] = left[i] * right[i]

        return nums

        # time O(n)
        # space O(n)

    def productExceptSelf(self, nums):
        n = len(nums)

        res = [1] * n
        for i in range(1, n):
            res[i] = res[i - 1] * nums[i - 1]

        right = 1
        for i in range(n - 1, -1, -1):
            res[i] = right * res[i]
            right *= nums[i]

        return res

        # time O(n)
        # space O(1) -> without the result


# -----------------------------------------------------------------------
"""
56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

Example 1:

Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6]
"""


class Solution:
    def merge(self, intervals):
        if len(intervals) < 2:
            return intervals
        intervals.sort(key=lambda pair: pair[0])
        res = []
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            last = res[-1]
            if last[1] >= intervals[i][0]:
                res[-1][1] = max(intervals[i][1], last[1])
            else:
                res.append(intervals[i])

        return res

        # time O(n*log(n))
        # space O(n)


# -----------------------------------------------------------------------
"""
The distance between 2 binary strings is the sum of their lengths after removing the common prefix. 
For example: the common prefix of 1011000 and 1011110 is 1011 so the distance is len("000") + len("110") = 3 + 3 = 6.

Given a list of binary strings, pick a pair that gives you maximum distance among all possible pair and return that distance.
"""


class Trie:
    def __init__(self):
        self.root = {}

    def add(self, s):
        curr = self.root
        for c in s:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]

        curr['*'] = True

    def print(self):
        curr = self.root
        self.recurPrint(curr, [])

    def recurPrint(self, node, curr):
        if not node or node == True:
            print(''.join(curr[:-1]))
            return

        for c in node:
            curr.append(c)
            self.recurPrint(node[c], curr)
            curr.pop()


def maxDistance(strings):
    tree = Trie()
    for s in strings:
        tree.add(s)
    curr = tree.root
    return getMaxDistance(curr)[1]

    # time O(n * k) -> n number of strings, k largest string
    # space O( n * k )


def getMaxDistance(node):
    if not node or node == '*' or node == True:
        return 0, 0  # -> (max depth, max distance)

    res = []
    for c in node:
        res.append(getMaxDistance(node[c]))

    currLen = 0
    distance = 0
    prevDistance = 0
    for pair in res:
        currLen = max(currLen, pair[0])
        distance += pair[0]
        prevDistance = max(prevDistance, pair[1])

    if len(res) < 2:
        distance = 0

    return currLen + 1, max(distance, prevDistance)


# -----------------------------------------------------------------------
"""
You are given 2 arrays representing integer locations of stores and houses (each location in this problem 
is one-dementional). For each house, find the store closest to it.
Return an integer array result where result[i] should denote the location of the store closest to the i-th house. 
If many stores are equidistant from a particular house, choose the store with the smallest numerical location. 
Note that there may be multiple stores and houses at the same location.

Example 1:

Input: houses = [5, 10, 17], stores = [1, 5, 20, 11, 16]
Output: [5, 11, 16]
Explanation: 
The closest store to the house at location 5 is the store at the same location.
The closest store to the house at location 10 is the store at the location 11.
The closest store to the house at location 17 is the store at the location 16.
Example 2:

Input: houses = [2, 4, 2], stores = [5, 1, 2, 3]
Output: [2, 3, 2]
Example 3:

Input: houses = [4, 8, 1, 1], stores = [5, 3, 1, 2, 6]
Output: [3, 6, 1, 1]
"""


def closestStore(houses, stores):
    if len(stores) < 1:
        return [sys.maxsize] * len(houses)

    stores = set(stores)
    stores = [val for val in stores]

    for i in range(len(houses)):
        houses[i] = getClosestStore(stores, houses[i])

    return houses


def getClosestStore(stores, house):
    start, end = 0, len(stores) - 1
    res = (stores[0], abs(stores[0] - house))

    while start <= end:
        mid = (start + end)
        store = stores[mid]
        distance = abs(house - store)

        if res[1] > distance:
            res = (store, distance)
        elif res[1] == distance:
            if res[0] > store:
                res = (store, distance)
        if store == house:
            return res[0]
        elif house < store:
            end = mid - 1
        else:
            start = mid + 1

    return res[0]


# -----------------------------------------------------------------------
"""
GCD (Greatest Common Divisor) of two positive integers is the largest positive integer that divides both numbers 
without a remainder.
Siblings: Nodes with the same parent are called siblings.
Level of a tree: Level of a tree is the number of edges on the longest path from the root node to a leaf.
You are given nodes of a binary tree of leven n as input.
Caluclate the GCD of each pair of siblings and then find the max & min GCD among them. 
Print the difference of max & min GCD ( max GCD - min GCD)

Note:
Print -1 if input tree is empty i.e level of tree is -1.
Consider those nodes which have a sibling
Print 0 if no such pair of siblings found
Input Format:
The input is in the following format:

The first line takes an integer n as input which represents the level of tree (the root node is at 0 level). 
(if level is equal to -1, means empty tree)
Next n+1 lines contain the nodes in the tree level order. Each i'th line represents the nodes present in the 
binary tree in i'th level.
1st line contains level 0 nodes. (i.e. root node).
2nd line contains nodes for level 1.
3rd line contains nodes for level 2 and so on.
Each node is represented by an integer value. Node value of -1 denotes an empty node(no node present at that place).

Output Format:
A single integer i.e., the difference of max & min GCD (max GCD - min GCD)

Constraints:
-1 <= level of tree <= 20
0 < element at nodes of tree <= 500
"""


def diffGCD():
    minGCD, maxGCD = sys.maxsize, -1 * sys.maxsize

    n = int(input())
    if n == -1:
        return -1

    for i in range(n + 1):
        if i == 0:
            input()  # --- for the root of the tree
            continue
        currLevel = list(map(int, input().split()))
        for j in range(0, pow(2, i), 2):
            first, second = currLevel[j], currLevel[j + 1]

            if first == -1 or second == -1:
                continue
            currGCD = gcd(first, second)

            if currGCD < minGCD:
                minGCD = currGCD

            if currGCD > maxGCD:
                maxGCD = currGCD

    if minGCD == sys.maxsize:
        return 0

    return maxGCD - minGCD


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


# -----------------------------------------------------------------------
"""
801. Minimum Swaps To Make Sequences Increasing

We have two integer sequences A and B of the same non-zero length.

We are allowed to swap elements A[i] and B[i].  Note that both elements are in the same index position in their respective sequences.

At the end of some number of swaps, A and B are both strictly increasing.  (A sequence is strictly increasing if and only if A[0] < A[1] < A[2] < ... < A[A.length - 1].)

Given A and B, return the minimum number of swaps to make both sequences strictly increasing.  It is guaranteed that the given input always makes it possible.

Example:
Input: A = [1,3,5,4], B = [1,2,3,7]
Output: 1
Explanation: 
Swap A[3] and B[3].  Then the sequences are:
A = [1, 3, 5, 7] and B = [1, 2, 3, 4]
which are both strictly increasing.

"""


class Solution:
    def minSwap(self, A, B):
        noSwap1, swap1 = 0, 1

        for i in range(1, len(A)):
            noSwap2 = swap2 = float('inf')

            if A[i - 1] < A[i] and B[i - 1] < B[i]:
                noSwap2 = min(noSwap2, noSwap1)
                swap2 = min(swap2, swap1 + 1)
            if A[i - 1] < B[i] and B[i - 1] < A[i]:
                noSwap2 = min(noSwap2, swap1)
                swap2 = min(swap2, noSwap1 + 1)

            noSwap1, swap1 = noSwap2, swap2

        return min(noSwap1, swap1)

        # time O(n)
        # space O(1)

# -----------------------------------------------------------------------
"""

"""
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
