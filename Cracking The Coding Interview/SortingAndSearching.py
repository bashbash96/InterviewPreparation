def BubbleSort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    # time O(n^2)
    # space O(1)


def selectionSort(arr):
    for i in range(len(arr)):
        minIdx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIdx]:
                minIdx = j

        if minIdx != i:
            arr[i], arr[minIdx] = arr[minIdx], arr[i]

    # time O(n^2)
    # space O(1)


def insertionSort(arr):
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] > arr[i]:
                arr[i], arr[j] = arr[j], arr[i]

    # time O(n^2)
    # space O(1)


def bucketSort(arr):
    if len(arr) == 0:
        return arr

    numOfBuckets = round(pow(len(arr), 0.5))
    buckets = [0] * numOfBuckets
    for i in range(len(buckets)):
        buckets[i] = []

    maxVal = max(arr)

    for i in range(len(arr)):
        currIdx = (arr[i] * numOfBuckets) // maxVal

        if arr[i] == maxVal:
            currIdx = len(buckets) - 1

        buckets[currIdx].append(arr[i])

    res = []
    for i in range(len(buckets)):
        buckets[i].sort()
        res += buckets[i]

    return res

    # time O(n log(n))
    # space O(n)


def mergeSort(arr):
    if len(arr) < 2:
        return
    mergeSort2(arr, 0, len(arr) - 1)

    # time O(n log(n))
    # space O(n)


def mergeSort2(arr, start, end):
    if start < end:
        mid = (start + end) // 2
        mergeSort2(arr, start, mid)
        mergeSort2(arr, mid + 1, end)
        merge(arr, start, end)


def merge(arr, start, end):
    mid = (start + end) // 2

    left, right = arr[start:mid + 1], arr[mid + 1:end + 1]

    l, r, k = 0, 0, start

    while l < len(left) and r < len(right):
        if right[r] < left[l]:
            arr[k] = right[r]
            r += 1
        else:
            arr[k] = left[l]
            l += 1
        k += 1

    while l < len(left):
        arr[k] = left[l]
        k += 1
        l += 1

    while r < len(right):
        arr[k] = right[r]
        k += 1
        r += 1


def quickSort(arr):
    if len(arr) < 2:
        return

    quickSort2(arr, 0, len(arr) - 1)

    # time O(n log(n))
    # space O(n)


def quickSort2(arr, start, end):
    if start < end:
        p = partition(arr, start, end)
        quickSort2(arr, start, p - 1)
        quickSort2(arr, p + 1, end)


def partition(arr, start, end):
    pivot = arr[start]

    left, right = start, end

    while left < right:
        while arr[right] > pivot and left < right:
            right -= 1

        while arr[left] <= pivot and left < right:
            left += 1

        arr[left], arr[right] = arr[right], arr[left]

    arr[left], arr[start] = arr[start], arr[left]

    return left


def countingSort(arr):
    k = max(arr) + 2
    counts = [0 for _ in range(k)]
    for val in arr:
        counts[val + 1] += 1

    for i in range(1, k):
        counts[i] += counts[i - 1]

    temp = [0] * len(arr)

    for val in arr:
        temp[counts[val]] = val
        counts[val] += 1

    arr = temp

    return arr

    # time O(n+k)
    # space O(k)


# -----------------------------------------------------------------------
"""
10.1 Sorted Merge: You are given two sorted arrays, A and B, where A has a large enough buffer at the
end to hold B. Write a method to merge B into A in sorted order.
Hints: #332
"""


def mergeSorted(arr1, arr2):
    p1, p2 = len(arr1) - len(arr2) - 1, len(arr2) - 1
    res = len(arr1) - 1

    while p1 >= 0 and p2 >= 0:
        if arr1[p1] > arr2[p2]:
            arr1[res] = arr1[p1]
            p1 -= 1
        elif arr2[p2] > arr1[p1]:
            arr1[res] = arr2[p2]
            p2 -= 1
        else:
            arr1[res] = arr2[p2]
            arr1[res - 1] = arr1[p1]
            res -= 1
            p1 -= 1
            p2 -= 1
        res -= 1

    while p2 >= 0:
        arr1[res] = arr2[p2]
        res -= 1
        p2 -= 1

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
10.2 Group Anagrams: Write a method to sort an array of strings so that all the anagrams are next to
each other.
"""


def sort(arr):
    map = {}

    for s in arr:
        currKey = ''.join(sorted(s))
        if currKey in map:
            map[currKey].append(s)
        else:
            map[currKey] = [s]

    i = 0
    for key in map:
        for val in map[key]:
            if i < len(arr):
                arr[i] = val
                i += 1

    return arr

    # time O(n*k*log(k)) -> k is the length of the longest string
    # space O(n*k)


# -----------------------------------------------------------------------
"""
10.3 Search in Rotated Array: Given a sorted array of n integers that has been rotated an unknown
number of times, write code to find an element in the array. You may assume that the array was
originally sorted in increasing order.
EXAMPLE
Inputfind 5 in {15, 16, 19, 20, 25, 1, 3,4,5,7,10, 14}
Output 8 (the index of 5 in the array)
"""


def searchRotatedArray(arr, val):
    rotateIdx = getRotateIdx(arr)

    if rotateIdx == -1:
        return binarySearch(arr, val, 0, len(arr) - 1)
    else:
        if val >= arr[0]:
            return binarySearch(arr, val, 0, rotateIdx)
        else:
            return binarySearch(arr, val, rotateIdx + 1, len(arr) - 1)


def binarySearch(arr, val, start, end):
    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == val:
            return mid
        elif val > arr[start]:
            start = mid + 1
        else:
            end = mid - 1

    return -1


def getRotateIdx(arr):
    start, end = 0, len(arr) - 1

    while start < end:
        mid = (start + end) // 2
        if arr[mid] > arr[mid + 1]:
            return mid
        elif arr[mid] >= arr[start]:
            start = mid + 1
        else:
            end = mid

    return -1


# -----------------------------------------------------------------------
"""
10.4 Sorted Search, No Size: You are given an array-like data structure Listy which lacks a size
method. It does, however, have an e lementAt (i) method that returns the element at index i in
0(1) time. If i is beyond the bounds of the data structure, it returns - 1. (For this reason, the data
structure only supports positive integers.) Given a Listy which contains sorted, positive integers,
find the index at which an element x occurs. If x occurs multiple times, you may return any index.
"""


def sortedSearch(arr, val):
    length = getLength(arr)

    start, end = 0, length - 1

    while start <= end:
        mid = (start + end) // 2
        if val > arr[mid]:
            start = mid + 1
        elif val < arr[mid]:
            end = mid - 1
        else:
            return mid

    return -1

    # time O(log(n))
    # space O(1)


def getLength(arr):
    length = 1

    while True:

        if arr[length - 1] != -1 and arr[length] == -1:
            return length
        elif arr[length - 1] != -1 and arr[length] != -1:
            length *= 2
        elif arr[length - 1] == -1 and arr[length - 2] != -1:
            return length - 1
        else:
            length = ((length // 2) + length) // 2


# -----------------------------------------------------------------------
"""
10.S Sparse Search: Given a sorted array of strings that is interspersed with empty strings, write a
method to find the location of a given string.
EXAMPLE
Input: ball, {"at",
U"}
Output: 4
"""


def sparseSearch(arr, s):
    start, end = 0, len(arr) - 1

    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == '':
            left = mid - 1
            right = mid + 1

            while arr[left] == '' and left > start:  # nearest left non empty str or till start
                left -= 1
            while arr[right] == '' and right < end:  # nearest right non empty str or till end
                right += 1
            # if the string is smaller than the left one then its in the left half
            if arr[left] != '' and s < arr[left]:
                end = left - 1
            # if the string is bigger than the right one then its in the right half
            elif arr[right] != '' and s > arr[right]:
                start = right + 1
            else:
                if arr[left] == s:
                    return left
                elif arr[right] == s:
                    return right
                else:
                    return -1
        else:
            if s > arr[mid]:
                start = mid + 1
            elif s < arr[mid]:
                end = mid - 1
            else:
                return mid

    return -1

    # time O(log(n))
    # space O(1)


# -----------------------------------------------------------------------
"""
10.9 Sorted Matrix Search: Given an M x N matrix in which each row and each column is sorted in
ascending order, write a method to find an element.
"""


def sortedMatrixSearch(mat, val):
    if val < mat[0][0] or val > mat[len(mat) - 1][len(mat[0]) - 1]:
        return False

    upRightRow, upRightCol = 0, len(mat[0]) - 1
    bottomLeftRow, bottomLeftCol = len(mat) - 1, 0

    while upRightCol != bottomLeftCol or upRightRow != bottomLeftRow:
        if val == mat[upRightRow][upRightCol] or val == mat[bottomLeftRow][bottomLeftCol]:
            return True

        if val > mat[upRightRow][upRightCol]:
            upRightRow += 1
        elif val < mat[upRightRow][upRightCol]:
            upRightCol -= 1

        if val > mat[bottomLeftRow][bottomLeftCol]:
            bottomLeftCol += 1
        elif val < mat[bottomLeftRow][bottomLeftRow]:
            bottomLeftRow -= 1

    if val == mat[upRightRow][upRightCol]:
        return True
    return False


# -----------------------------------------------------------------------
"""
10.10 Rank from Stream: Imagine you are reading in a stream of integers. Periodically, you wish
to be able to look up the rank of a number x (the number of values less than or equal to x).
Implement the data structures and algorithms to support these operations. That is, implement
the method track(int x), which is called when each number is generated, and the method
getRankOfNumber(int x), which returns the number of values less than or equal to x (not
including x itself).
EXAMPLE
Stream (in order of appearance): 5 , 1 , 4 , 4 , 5 , 9 , 7 , 13 , 3
getRankOfNumber(l) 0
getRankOfNumber(3) 1
getRankOfNumber(4) 3
"""


class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.rank = 0


class RankedBST:
    def __init__(self):
        self.root = None

    def getRank(self, x):

        return self.recurGetRank(self.root, x, 0)

        # time O(log(n))
        # space O(log(n))

    def recurGetRank(self, node, x, counter):
        if not node:
            return -1

        if x == node.data:
            return counter + node.rank
        elif x > node.data:
            return self.recurGetRank(node.right, x, counter + node.rank + 1)
        else:
            return self.recurGetRank(node.left, x, counter)

    def track(self, x):
        if not self.root:
            self.root = Node(x)
        else:

            self.recurTrack(self.root, Node(x))

        # time O(log(n))
        # space O(log(n))

    def recurTrack(self, node, newNode):
        x = newNode.data

        if x > node.data:
            if not node.right:
                node.right = newNode
            else:
                self.recurTrack(node.right, newNode)
        else:
            node.rank += 1
            if not node.left:
                node.left = newNode
            else:
                self.recurTrack(node.left, newNode)


# -----------------------------------------------------------------------
"""
10.11 Peaks and Valleys: In an array of integers, a "peak" is an element which is greater than or equal
to the adjacent integers and a "valley" is an element which is less than or equal to the adjacent
integers. For example, in the array {S, 8, 6, 2, 3, 4, 6}, {8, 6} are peaks and {S, 2} are valleys. Given an
array of integers, sort the array into an alternating sequence of peaks and valleys.
EXAMPLE
Input: {S, 3, 1,2, 3}
Output: {S, 1,3,2, 3}
"""


def peaksAndValleys(arr):
    arr = sorted(arr)

    for i in range(1, len(arr) - 1, 2):
        arr[i], arr[i + 1] = arr[i + 1], arr[i]

    return arr

    # time O(n*log(n))
    # space O(1)


def peaksAndValleys2(arr):
    if len(arr) < 3:
        return arr
    for i in range(1, len(arr), 2):
        maxIdx = getBiggestIdx(arr, i - 1, i, i + 1)
        if maxIdx != i:
            arr[i], arr[maxIdx] = arr[maxIdx], arr[i]

    return arr

    # time O(n)
    # space O(1)


def getBiggestIdx(arr, prev, curr, next):
    maxIdx = curr
    if arr[prev] > arr[maxIdx]:
        maxIdx = prev

    if next < len(arr) and arr[next] > arr[maxIdx]:
        maxIdx = next

    return maxIdx
