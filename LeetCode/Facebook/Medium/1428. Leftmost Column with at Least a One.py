"""
(This problem is an interactive problem.)

A row-sorted binary matrix means that all elements are 0 or 1 and each row of the matrix is sorted in non-decreasing order.

Given a row-sorted binary matrix binaryMatrix, return the index (0-indexed) of the leftmost column with a 1 in it. If such an index does not exist, return -1.

You can't access the Binary Matrix directly. You may only access the matrix using a BinaryMatrix interface:

BinaryMatrix.get(row, col) returns the element of the matrix at index (row, col) (0-indexed).
BinaryMatrix.dimensions() returns the dimensions of the matrix as a list of 2 elements [rows, cols], which means the matrix is rows x cols.
Submissions making more than 1000 calls to BinaryMatrix.get will be judged Wrong Answer. Also, any solutions that attempt to circumvent the judge will result in disqualification.

For custom testing purposes, the input will be the entire binary matrix mat. You will not have access to the binary matrix directly.



Example 1:



Input: mat = [[0,0],[1,1]]
Output: 0
Example 2:



Input: mat = [[0,0],[0,1]]
Output: 1
Example 3:



Input: mat = [[0,0],[0,0]]
Output: -1
Example 4:



Input: mat = [[0,0,0,1],[0,0,1,1],[0,1,1,1]]
Output: 1


Constraints:

rows == mat.length
cols == mat[i].length
1 <= rows, cols <= 100
mat[i][j] is either 0 or 1.
mat[i] is sorted in non-decreasing order.
"""


# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
# class BinaryMatrix(object):
#    def get(self, row, col):
#        """
#        :type row : int, col : int
#        :rtype int
#        """
#
#    def dimensions:
#        """
#        :rtype list[]
#        """

class Solution(object):
    def leftMostColumnWithOne(self, binaryMatrix):
        """
        :type binaryMatrix: BinaryMatrix
        :rtype: int
        """

        rows, cols = binaryMatrix.dimensions()

        #         row, col = 0, cols - 1

        #         while row < rows and col >= 0:

        #             while col >= 0 and binaryMatrix.get(row, col) == 1:
        #                 col -= 1

        #             row += 1

        #         return -1 if col == cols - 1 else col + 1

        #     # time O(rows + cols)
        #     # space O(1)

        res = -1

        for row in range(rows):
            if res == -1:
                res = get_left_most_col(binaryMatrix, row, cols - 1)
            else:
                curr = get_left_most_col(binaryMatrix, row, res)
                if curr == -1:
                    continue
                res = min(res, curr)

        return res

    # time O(rows * log(cols))
    # space O(1)


def get_left_most_col(binaryMatrix, row, res):
    if res == 0:
        return res

    left, right = 0, res
    ans = -1

    while left <= right:
        mid = (left + right) >> 1

        if binaryMatrix.get(row, mid) == 1:
            ans = mid
            right = mid - 1
        else:
            left = mid + 1

    return ans


"""
1:
for each row get the left most column with 1 using binary search.
maintain the leftmost column as result.

=> time O(rows * log(cols))
=> space O(1)

2:
for each row get the leftmost column depends on the result till now.

=> time O(rows + cols)
=> space O(1)


"""
