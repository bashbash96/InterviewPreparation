# -----------------------------------------------------------------------
"""
Intersection Of Sorted Arrays

Find the intersection of two sorted arrays.
OR in other words,
Given 2 sorted arrays, find all the elements which occur in both the arrays.

Example :

Input :
    A : [1 2 3 3 4 5 6]
    B : [3 3 5]

Output : [3 3 5]

Input :
    A : [1 2 3 3 4 5 6]
    B : [3 5]

Output : [3 5]
"""


class Solution:
    # @param A : tuple of integers
    # @param B : tuple of integers
    # @return a list of integers
    def intersect(self, A, B):
        if len(A) > len(B):
            return self.intersect(B, A)

        res = []
        i, j = 0, 0
        while i < len(A) and j < len(B):
            if A[i] == B[j]:
                res.append(A[i])
                i += 1
                j += 1
            elif A[i] < B[j]:
                i += 1
            else:
                j += 1

        return res

    # time O(n + m)
    # space O(min(n, m))


# -----------------------------------------------------------------------
"""
Remove Duplicates from Sorted Array

Remove duplicates from Sorted Array
Given a sorted array, remove the duplicates in place such that each element appears only once and return the new length.

Note that even though we want you to return the new length, make sure to change the original array as well in place

Do not allocate extra space for another array, you must do this in place with constant memory.
"""


class Solution:
    # @param A : list of integers
    # @return an integer
    def removeDuplicates(self, A):
        length, i = 0, 0

        while i < len(A):
            if i < len(A) - 1 and A[i] == A[i + 1]:
                i += 1
                continue
            else:
                A[length] = A[i]
                length += 1
            i += 1
        return length

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
Array 3 Pointers

You are given 3 arrays A, B and C. All 3 of the arrays are sorted.

Find i, j, k such that :
max(abs(A[i] - B[j]), abs(B[j] - C[k]), abs(C[k] - A[i])) is minimized.
Return the minimum max(abs(A[i] - B[j]), abs(B[j] - C[k]), abs(C[k] - A[i]))

**abs(x) is absolute value of x and is implemented in the following manner : **

      if (x < 0) return -x;
      else return x;
Example :

Input : 
        A : [1, 4, 10]
        B : [2, 15, 20]
        C : [10, 12]

Output : 5 
         With 10 from A, 15 from B and 10 from C. 
"""


class Solution:
    # @param A : tuple of integers
    # @param B : tuple of integers
    # @param C : tuple of integers
    # @return an integer
    def minimize(self, A, B, C):

        p1, p2, p3 = 0, 0, 0

        min_res = float('inf')

        while p1 < len(A) and p2 < len(B) and p3 < len(C):
            curr = max(abs(A[p1] - B[p2]), abs(B[p2] - C[p3]), abs(C[p3] - A[p1]))

            if curr <= min_res:
                min_res = curr

            min_val = min(A[p1], B[p2], C[p3])

            if A[p1] == min_val:
                p1 += 1
            elif B[p2] == min_val:
                p2 += 1
            else:
                p3 += 1

        return min_res

    # time O(min(n1, n2, n3))
    # space O(1)

# -----------------------------------------------------------------------
