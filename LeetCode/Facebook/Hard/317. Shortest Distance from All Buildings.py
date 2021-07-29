"""
You are given an m x n grid grid of values 0, 1, or 2, where:

each 0 marks an empty land that you can pass by freely,
each 1 marks a building that you cannot pass through, and
each 2 marks an obstacle that you cannot pass through.
You want to build a house on an empty land that reaches all buildings in the shortest total travel distance. You can only move up, down, left, and right.

Return the shortest travel distance for such a house. If it is not possible to build such a house according to the above rules, return -1.

The total travel distance is the sum of the distances between the houses of the friends and the meeting point.

The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.



Example 1:


Input: grid = [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]
Output: 7
Explanation: Given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2).
The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is minimal.
So return 7.
Example 2:

Input: grid = [[1,0]]
Output: 1
Example 3:

Input: grid = [[1]]
Output: -1


Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 100
grid[i][j] is either 0, 1, or 2.
There will be at least one building in the grid.
"""

from collections import defaultdict


class Solution(object):
    def shortestDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        empty_lands, buildings = get_empty_and_buildings(grid)

        distances = defaultdict(list)
        res = float('inf')

        for point in buildings:
            bfs(grid, point, distances)

        for point in empty_lands:
            if len(distances[point]) == len(buildings):
                res = min(res, sum(distances[point]))

        return res if res != float('inf') else -1

    # time O(e * k), e: num of empty, k: num of buildings
    # space O(e * k)


def bfs(grid, source, distances):
    curr = source
    visited = set()

    queue = deque()
    queue.append((curr, 0))

    while queue:
        curr, dist = queue.popleft()

        row, col = curr
        if grid[row][col] == 0:
            distances[curr].append(dist)

        for n_row, n_col in get_neighbors(row, col):
            if not is_valid(grid, n_row, n_col):
                continue

            if (n_row, n_col) in visited:
                continue

            visited.add((n_row, n_col))
            queue.append(((n_row, n_col), dist + 1))


def get_empty_and_buildings(grid):
    buildings = set()
    empty = set()

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 0:
                empty.add((row, col))
            elif grid[row][col] == 1:
                buildings.add((row, col))

    return empty, buildings


def is_valid(grid, row, col):
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == 2 or grid[row][col] == 1:
        return False

    return True


def get_neighbors(row, col):
    dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    res = []

    for dx, dy in dirs:
        res.append((row + dx, col + dy))

    return res


"""
[[1,0,2,0,1]
,[0,0,0,0,0]
,[0,0,1,0,0]]



"""
