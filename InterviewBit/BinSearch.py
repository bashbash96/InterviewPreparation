# -----------------------------------------------------------------------
"""
Search for a Range

Given a sorted array of integers A(0 based index) of size N, find the starting and ending position of a given integar B in array A.

Your algorithm’s runtime complexity must be in the order of O(log n).

Return an array of size 2, such that first element = starting position of B in A and second element = ending position of B in A, if B is not found in A return [-1, -1].



Input Format

The first argument given is the integer array A.
The second argument given is the integer B.
Output Format

 Return an array of size 2, such that first element = starting position of B in A and second element = ending position of B in A, if B is not found in A return [-1, -1].
Constraints

1 <= N <= 10^6
1 <= A[i], B <= 10^9
For Example

Input 1:
    A = [5, 7, 7, 8, 8, 10]
    B = 8
Output 1:
    [3, 4]
Explanation 1:
    First occurence of 8 in A is at index 3
    Second occurence of 8 in A is at index 4
    ans = [3, 4]

Input 2:
    A = [5, 17, 100, 111]
    B = 3
Output 2:
    [-1, -1]
"""


class Solution:
    # @param A : tuple of integers
    # @param B : integer
    # @return a list of integers
    def searchRange(self, A, B):
        left = get_limit(A, B, 'left')
        right = get_limit(A, B, 'right')

        return [left, right]

    # time O(log(n))
    # space O(1)


def get_limit(arr, num, limit):
    start, end = 0, len(arr) - 1
    res = -1
    while start <= end:
        mid = (start + end) // 2
        if num < arr[mid]:
            end = mid - 1
        elif num > arr[mid]:
            start = mid + 1
        else:
            res = mid
            if limit == 'left':
                end = mid - 1
            else:
                start = mid + 1

    return res


# -----------------------------------------------------------------------
"""
Implement Power Function

Implement pow(x, n) % d.

In other words, given x, n and d,

find (xn % d)

Note that remainders on division cannot be negative.
In other words, make sure the answer you return is non negative.

Input : x = 2, n = 3, d = 3
Output : 2

2^3 % 3 = 8 % 3 = 2.
"""


class Solution:
    # @param x : integer
    # @param n : integer
    # @param d : integer
    # @return an integer
    def pow(self, x, n, d):

        if n == 0:
            if x == 0:
                return 0
            return 1

        if d == 1:
            return 0

        calc = self.pow(x, n // 2, d)
        if n % 2 == 0:
            return (calc * calc) % d
        else:
            return (x * calc * calc) % d

    # time O(log(n))
    # space O(log(n))


# -----------------------------------------------------------------------
"""
Rotated Sorted Array Search

Given an array of integers A of size N and an integer B.

array A is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2 ).

You are given a target value B to search. If found in the array, return its index, otherwise return -1.

You may assume no duplicate exists in the array.

NOTE:- Array A was sorted in non-decreasing order before rotation.

NOTE : Think about the case when there are duplicates. Does your current solution work? How does the time complexity change?*
Input Format

The first argument given is the integer array A.
The second argument given is the integer B.
Output Format

Return index of B in array A, otherwise return -1
Constraints

1 <= N <= 1000000
1 <= A[i] <= 10^9
all elements in A are disitinct.
For Example

Input 1:
    A = [4, 5, 6, 7, 0, 1, 2, 3]
    B = 4
Output 1:
    0
Explanation 1:
 Target 4 is found at index 0 in A.


Input 2:
    A = [5, 17, 100, 3]
    B = 6
Output 2:
    -1
"""


class Solution:
    # @param A : tuple of integers
    # @param B : integer
    # @return an integer
    def search(self, A, B):
        arr = list(A)
        if not arr:
            return -1

        target = B
        pivot = get_pivot(arr)
        if pivot == -1:
            return bin_search(arr, 0, len(arr) - 1, target)

        if target >= arr[0] and target <= arr[pivot]:
            return bin_search(arr, 0, pivot, target)

        return bin_search(arr, pivot + 1, len(arr) - 1, target)

    # time O(log(n))
    # space O(1)


def bin_search(arr, start, end, target):
    while start <= end:
        mid = (start + end) // 2
        if target < arr[mid]:
            end = mid - 1
        elif target > arr[mid]:
            start = mid + 1
        else:
            return mid

    return -1


def get_pivot(arr):
    start, end = 0, len(arr) - 1
    if len(arr) < 2:
        return -1

    while start <= end:
        mid = (start + end) // 2

        if mid < len(arr) - 1 and arr[mid] > arr[mid + 1]:
            return mid
        elif arr[mid] >= arr[start]:
            start = mid + 1
        else:
            end = mid - 1

    return -1


# -----------------------------------------------------------------------
"""
Allocate Books

Given an array of integers A of size N and an integer B.

College library has N bags,the ith book has A[i] number of pages.

You have to allocate books to B number of students so that maximum number of pages alloted to a student is minimum.

A book will be allocated to exactly one student.
Each student has to be allocated at least one book.
Allotment should be in contiguous order, for example: A student cannot be allocated book 1 and book 3, skipping book 2.
Calculate and return that minimum possible number.

NOTE: Return -1 if a valid assignment is not possible.



Input Format

The first argument given is the integer array A.
The second argument given is the integer B.
Output Format

Return that minimum possible number
Constraints

1 <= N <= 10^5
1 <= A[i] <= 10^5
For Example

Input 1:
    A = [12, 34, 67, 90]
    B = 2
Output 1:
    113
Explanation 1:
    There are 2 number of students. Books can be distributed in following fashion : 
        1) [12] and [34, 67, 90]
        Max number of pages is allocated to student 2 with 34 + 67 + 90 = 191 pages
        2) [12, 34] and [67, 90]
        Max number of pages is allocated to student 2 with 67 + 90 = 157 pages 
        3) [12, 34, 67] and [90]
        Max number of pages is allocated to student 1 with 12 + 34 + 67 = 113 pages

        Of the 3 cases, Option 3 has the minimum pages = 113.

Input 2:
    A = [5, 17, 100, 11]
    B = 4
Output 2:
    100
"""


class Solution:
    # @param A : list of integers
    # @param B : integer
    # @return an integer
    def books(self, A, B):
        arr = A
        students_num = B

        n = len(arr)
        # return -1 if no. of books is
        # less than no. of students
        if (n < students_num):
            return -1

        curr_sum = 0
        # Count total number of pages
        for val in arr:
            curr_sum += val

        # initialize start as 0 pages and
        # end as total pages
        start, end = 0, curr_sum
        min_pages = float('inf')

        # traverse until start <= end
        while (start <= end):

            # check if it is possible to distribute
            # books by using mid as current minimum
            mid_sum = (start + end) // 2
            if is_possible(arr, students_num, mid_sum):

                # if yes then find the minimum distribution
                min_pages = min(min_pages, mid_sum)

                # as we are finding minimum and books
                # are sorted so reduce end = mid -1
                end = mid_sum - 1

            else:
                # if not possible means pages should be
                # increased so update start = mid + 1
                start = mid_sum + 1

        # at-last return minimum no. of pages
        return min_pages

    # time O(n * log(m)) -> n: num of books, m: num of total pages
    # space O(1)


def is_possible(arr, students_num, curr_min):
    n = len(arr)
    studentsRequired = 1
    curr_sum = 0

    # iterate over all books
    for i in range(n):

        # check if current number of pages are
        # greater than curr_min that means
        # we will get the result after
        # mid no. of pages
        if (arr[i] > curr_min):
            return False

        # count how many students are required
        # to distribute curr_min pages
        if curr_sum + arr[i] > curr_min:
            # increment student count
            studentsRequired += 1

            # update curr_sum
            curr_sum = arr[i]

            # if students required becomes greater
            # than given no. of students, return False
            if studentsRequired > students_num:
                return False

        # else update curr_sum
        else:
            curr_sum += arr[i]

    return True


# -----------------------------------------------------------------------
"""
Painter's Partition Problem

Given 2 integers A and B and an array of integars C of size N.

Element C[i] represents length of ith board.

You have to paint all N boards [C0, C1, C2, C3 … CN-1]. There are A painters available and each of them takes B units of time to paint 1 unit of board.

Calculate and return minimum time required to paint all boards under the constraints that any painter will only paint contiguous sections of board.

2 painters cannot share a board to paint. That is to say, a board
cannot be painted partially by one painter, and partially by another.
A painter will only paint contiguous boards. Which means a
configuration where painter 1 paints board 1 and 3 but not 2 is
invalid.
Return the ans % 10000003



Input Format

The first argument given is the integer A.
The second argument given is the integer B.
The third argument given is the integer array C.
Output Format

Return minimum time required to paint all boards under the constraints that any painter will only paint contiguous sections of board % 10000003.
Constraints

1 <=A <= 1000
1 <= B <= 10^6
1 <= C.size() <= 10^5
1 <= C[i] <= 10^6
For Example

Input 1:
    A = 2
    B = 5
    C = [1, 10]
Output 1:
    50
Explanation 1:
    Possibility 1:- same painter paints both blocks, time taken = 55units
    Possibility 2:- Painter 1 paints block 1, painter 2 paints block 2, time take = max(5, 50) = 50
    There are no other distinct ways to paint boards.
    ans = 50%10000003

Input 2:
    A = 10
    B = 1
    C = [1, 8, 11, 3]
Output 2:
    11
"""

MOD = 10000003


class Solution:
    # @param A : integer
    # @param B : integer
    # @param C : list of integers
    # @return an integer
    def paint(self, A, B, C):

        num_of_painters = A
        time_per_unit = B
        boards = [val * time_per_unit for val in C]

        max_time = sum(boards)

        left, right = 0, max_time
        min_time = float('inf')

        while left <= right:
            mid_time = (left + right) // 2

            if is_paintable(boards, num_of_painters, mid_time):
                min_time = min(min_time, mid_time)

                right = mid_time - 1
            else:
                left = mid_time + 1

        return min_time % MOD

    # time O(n * log(m))
    # space O(1)


def is_paintable(boards, num_of_painters, curr_min):
    curr_sum = 0
    painters_needed = 1

    for val in boards:

        if val > curr_min:
            return False

        if curr_sum + val > curr_min:
            curr_sum = val
            painters_needed += 1

            if painters_needed > num_of_painters:
                return False
        else:
            curr_sum += val

    return True


# -----------------------------------------------------------------------
"""
Find a peak element

Given an array of integers A, find and return the peak element in it.
An array element is peak if it is NOT smaller than its neighbors.
For corner elements, we need to consider only one neighbor.
For example, for input array {5, 10, 20, 15}, 20 is the only peak element.

Following corner cases give better idea about the problem.

1) If input array is sorted in strictly increasing order, the last element is always a peak element. 
For example, 5 is peak element in {1, 2, 3, 4, 5}.
2) If input array is sorted in strictly decreasing order, the first element is always a peak element. 
10 is the peak element in {10, 9, 8, 7, 6}.
Note: It is guranteed that the answer is unique.



Input Format

The only argument given is the integer array A.
Output Format

Return the peak element.
Constraints

1 <= length of the array <= 100000
1 <= A[i] <= 10^9 
For Example

Input 1:
    A = [1, 2, 3, 4, 5]
Output 1:
    5

Input 2:
    A = [5, 17, 100, 11]
Output 2:
    100
"""


class Solution:
    # @param A : list of integers
    # @return an integer
    def solve(self, A):
        arr = A
        n = len(arr)

        if n < 1:
            return -1

        if is_peak(arr, 0):
            return arr[0]
        if is_peak(arr, n - 1):
            return arr[n - 1]

        left, right = 1, n - 2

        while left <= right:
            mid = (left + right) // 2
            if is_peak(arr, mid):
                return arr[mid]

            if arr[mid - 1] > arr[mid]:
                right = mid - 1
            else:
                left = mid + 1

        return -1

    # time O(log(n))
    # space O(1)


def is_peak(arr, idx):
    if idx > 0:
        prev = arr[idx - 1]
    else:
        prev = float('-inf')

    if idx < len(arr) - 1:
        next_ = arr[idx + 1]
    else:
        next_ = float('-inf')

    return arr[idx] >= prev and arr[idx] >= next_

# -----------------------------------------------------------------------
