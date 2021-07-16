"""
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.



Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3


Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] is '0' or '1'.
"""


class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        islands = 0
        visited = set()

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '0':
                    continue

                if (row, col) not in visited:
                    visit_island(grid, row, col, visited)
                    islands += 1
        return islands

    # time O(n * m)
    # space O(n * m)


def visit_island(grid, row, col, visited):
    if (row, col) in visited or not is_valid(grid, row, col):
        return

    visited.add((row, col))

    for n_row, n_col in get_neighbors(row, col):
        visit_island(grid, n_row, n_col, visited)


def is_valid(grid, row, col):
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != '1':
        return False

    return True


def get_neighbors(row, col):
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    res = []

    for dx, dy in directions:
        res.append((row + dx, col + dy))

    return res
