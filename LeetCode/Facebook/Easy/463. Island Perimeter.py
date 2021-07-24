"""
You are given row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water.

Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.



Example 1:


Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
Output: 16
Explanation: The perimeter is the 16 yellow stripes in the image above.
Example 2:

Input: grid = [[1]]
Output: 4
Example 3:

Input: grid = [[1,0]]
Output: 4


Constraints:

row == grid.length
col == grid[i].length
1 <= row, col <= 100
grid[i][j] is 0 or 1.
"""


class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        row, col = get_island_idx(grid)

        if row == -1:
            return 0

        return visit_island(grid, row, col, set())

    # time O(rows * cols)
    # space O(rows * cols)


def get_island_idx(grid):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 1:
                return row, col

    # no island
    return -1, -1


def visit_island(grid, row, col, visited):
    curr_peri = 0

    if (row, col) in visited:
        return 0

    visited.add((row, col))

    for n_row, n_col in get_neighbors(row, col):

        if not is_valid(grid, n_row, n_col):
            curr_peri += 1
            continue

        curr_peri += visit_island(grid, n_row, n_col, visited)

    return curr_peri


def get_neighbors(row, col):
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    res = []

    for dx, dy in directions:
        res.append((row + dx, col + dy))

    return res


def is_valid(grid, row, col):
    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or grid[row][col] == 0:
        return False

    return True
