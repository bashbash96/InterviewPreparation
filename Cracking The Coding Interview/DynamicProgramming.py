# -----------------------------------------------------------------------
"""
8.1 Triple Step: A child is running up a staircase with n steps and can hop either 1 step, 2 steps, or 3
steps at a time. Implement a method to count how many possible ways the child can run up the
stairs.
"""


def triple_step(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 4

    prev1, prev2, prev3 = 4, 2, 1

    for _ in range(4, n + 1):
        curr = prev1 + prev2 + prev3
        prev3 = prev2
        prev2 = prev1
        prev1 = curr

    return curr

    # time O(n)
    # space O(1)


def tripleStep(n):
    memo = [None] * (n + 1)

    return recurTripleStep(n, memo)

    # time O(n)
    # space O(n)


def recurTripleStep(n, memo):
    if n < 0:
        return 0

    if n == 0:
        return 1

    if not memo[n]:
        memo[n] = recurTripleStep(n - 1, memo) + recurTripleStep(n - 2, memo) + recurTripleStep(n - 3, memo)

    return memo[n]


# -----------------------------------------------------------------------
"""
8.2 Robot in a Grid: Imagine a robot sitting on the upper left corner of grid with r rows and c columns.
The robot can only move in two directions, right and down, but certain cells are "off limits" such that
the robot cannot step on them. Design an algorithm to find a path for the robot from the top left to
the bottom right.
"""


def findPath(arr):
    memo = [[None for j in range(len(arr[0]))] for i in range(len(arr))]
    path = []
    recurFindPath(arr, len(arr) - 1, len(arr[0]) - 1, memo, path)
    return path

    # time O(n*m)
    # space O(n*m)


def recurFindPath(arr, row, col, memo, path):
    if arr[row][col] == 0 or row < 0 or col < 0:
        return False

    if row == 0 and col == 0:
        path.append((row, col))
        memo[row][col] = True

    if not memo[row][col]:
        memo[row][col] = recurFindPath(arr, row - 1, col, memo, path) or recurFindPath(arr, row, col - 1, memo, path)
        if memo[row][col]:
            path.append((row, col))

    return memo[row][col]


# -----------------------------------------------------------------------
"""
8.3 Magic Index: A magic index in an array A [1. .. n -1] is defined to be an index such that A[ i]
i. Given a sorted array of distinct integers, write a method to find a magic index, if one exists, in
array A.
FOLLOW UP
What if the values are not distinct?
"""


def magicIdx(arr):
    start, end = 0, len(arr) - 1

    while start <= end:
        mid = (start + end) // 2

        if mid == arr[mid]:
            return mid
        elif mid < arr[mid]:
            end = mid - 1
        else:
            start = mid + 1

    return -1

    # time O(log(n))
    # space O(1)


def magicIdx2(arr):
    return recurMagixIdx(arr, 0, len(arr) - 1)

    # time O(n)
    # space O(n)


def recurMagixIdx(arr, start, end):
    if end < start:
        return -1

    mid = (start + end) // 2
    midVal = arr[mid]

    if mid == midVal:
        return mid

    leftIdx = min(mid - 1, midVal)

    left = recurMagixIdx(arr, start, leftIdx)

    if left >= 0:
        return left

    rightIdx = max(mid + 1, midVal)

    right = recurMagixIdx(arr, rightIdx, end)

    return right


# -----------------------------------------------------------------------
"""
8.4 Power Set: Write a method to return all subsets of a set.
"""


def powerSet(arr):
    memo = {}

    return recurPowerSet(arr, len(arr) - 1, memo)

    # time O(n*2^n)
    # space O(n*2^n)


def recurPowerSet(arr, idx, memo):
    if idx < 0:
        return [[]]

    if idx not in memo:
        prev = recurPowerSet(arr, idx - 1, memo)
        currRes = []

        for val in prev:
            currRes.append([v for v in val])
            curr = [v for v in val]
            curr.append(arr[idx])
            currRes.append(curr)

        memo[idx] = currRes

    return memo[idx]


# -----------------------------------------------------------------------
"""
8.S Recursive Multiply: Write a recursive function to multiply two positive integers without using
the * operator (or / operator). You can use addition, subtraction, and bit shifting, but you should
minimize the number of those operations.
"""


def mult(n, m):
    if n == 0 or m == 0:
        return 0

    if n < 0 and m < 0:
        m *= -1
        n *= -1

    if n < 0 or m < 0:
        if n < 0:
            n *= -1
        elif m < 0:
            m *= -1

        if m > n:
            return mult(m, n)
        return -1 * recursiveMult(n, m)

    if m > n:
        return mult(m, n)

    return recursiveMult(n, m)

    # time O(log(s)) -> s is the smaller number
    # space O(log(s))


def recursiveMult(bigger, smaller):
    if smaller == 0:
        return 0

    if smaller == 1:
        return bigger

    half = smaller >> 1
    first = recursiveMult(bigger, half)
    second = first
    if smaller % 2 == 1:
        second += bigger

    return second + first


# -----------------------------------------------------------------------
"""
8.6 Towers of Hanoi: In the classic problem of the Towers of Hanoi, you have 3 towers and N disks of
different sizes which can slide onto any tower. The puzzle starts with disks sorted in ascending order
of size from top to bottom (Le., each disk sits on top of an even larger one). You have the following
constraints:
(1) Only one disk can be moved at a time.
(2) A disk is slid off the top of one tower onto another tower.
(3) A disk cannot be placed on top of a smaller disk.
Write a program to move the disks from the first tower to the last using Stacks.
"""


def towerOfHanoi(n):
    towers = 3 * [[]]
    recurTowerOfHanoi(n, 'a', 'c', 'b', towers)

    # time O(2^n)
    # space O(n)


def recurTowerOfHanoi(n, source, destination, buffer, towers):
    if n <= 0:
        return

    recurTowerOfHanoi(n - 1, source, buffer, destination, towers)
    print("move {} from {} to {} ".format(n, source, destination))
    recurTowerOfHanoi(n - 1, buffer, destination, source, towers)


# -----------------------------------------------------------------------
"""
8.7 Permutations without Dups: Write a method to compute all permutations of a string of unique
characters.
"""


def permutations(s):
    return recurPermutations(s, len(s) - 1)

    # time O(n^2*n!)
    # space O(n!)


def recurPermutations(s, idx):
    if idx == 0:
        return [s[idx]]

    prev = recurPermutations(s, idx - 1)  # O(n)

    currRes = []
    for val in prev:  # O(n!)
        for i in range(len(val) + 1):  # O(n)
            currRes.append(val[0:i] + s[idx] + val[i:])

    return currRes


# -----------------------------------------------------------------------
"""
8.8 Permutations with Duplicates: Write a method to compute all permutations of a string whose
characters are not necessarily unique. The list of permutations should not have duplicates.
"""


def permWithDup(s):
    count = countChars(s)
    res = []

    recurPermWithDup(count, [], len(s), res)
    return res

    # time O(n!\(k!)) -> k number of duplicated chars "each duplicated char"
    # space O(n!\(k!))


def countChars(s):
    count = {}

    for char in s:
        if char in count:
            count[char] += 1
        else:
            count[char] = 1

    return count


def recurPermWithDup(count, prefix, remaining, res):
    if remaining == 0:
        res.append(''.join(prefix))
        return

    for c in count:
        if count[c] > 0:
            count[c] -= 1
            recurPermWithDup(count, prefix + [c], remaining - 1, res)
            count[c] += 1


# -----------------------------------------------------------------------
"""
8.9 Parens: Implement an algorithm to print all valid (Le., properly opened and closed) combinations
of n pairs of parentheses.
EXAMPLE
Input: 3
Output: ( ( () ) ) , (() ()) , (() () ) , () ( () ) , () () ()
"""


def parens(n):
    if n <= 0:
        return ''

    str = [''] * (2 * n)
    res = []

    recurParens(res, str, n, n, 0)

    return res

    # time O(4^n/n^(0.5))
    # space O(4^n/n^(0.5))


def recurParens(res, str, leftRemain, rightRemain, idx):
    if leftRemain == 0 and rightRemain == 0:
        res.append(''.join(str))
        return

    elif leftRemain < 0 or rightRemain < leftRemain:
        return
    else:
        str[idx] = '('
        recurParens(res, str, leftRemain - 1, rightRemain, idx + 1)

        str[idx] = ')'
        recurParens(res, str, leftRemain, rightRemain - 1, idx + 1)


# -----------------------------------------------------------------------
"""
8.11 Coins: Given an infinite number of quarters (25 cents), dimes (10 cents), nickels (5 cents), and
pennies (1 cent), write code to calculate the number of ways of representing n cents.
"""


def coins(cents):
    coins = [1, 25, 10, 5]
    coins = [1, 2, 3]
    # return recur_coins(cents, coins, {}, len(coins)  - 1)

    memo = [0] * (cents + 1)

    memo[0] = 1
    for coin in coins:
        for amount in range(coin, cents + 1):
            memo[amount] += memo[amount - coin]

    return memo[cents]

    # time O(n*c) c -> num of coins
    # space O(n)


def recur_coins(amount, coins, memo, idx):
    if amount < 0:
        return 0

    if amount == 0:
        return 1

    if amount in memo:
        return memo[amount]

    if idx < 0:
        return 0

    curr = recur_coins(amount, coins, memo, idx - 1) + recur_coins(amount - coins[idx], coins, memo, idx)

    memo[amount] = curr
    return curr


# -----------------------------------------------------------------------
"""
8.12 Eight Queens: Write an algorithm to print all ways of arranging eight queens on an 8x8 chess board
so that none of them share the same row, column, or diagonal. In this case, "diagonal" means all
diagonals, not just the two that bisect the board.
"""


def queens(size):
    board = [[False for col in range(size)] for row in range(size)]

    if not placeQueen(board, col=0):
        return False, [[]]

    return True, board


def placeQueen(board, col):
    if col >= len(board):
        return True

    for row in range(len(board)):
        if isValid(board, row, col):

            board[row][col] = True

            if placeQueen(board, col + 1):
                return True

            board[row][col] = False

    return False


def isValid(board, row, col):
    # check all previous values in current row
    for i in range(col):
        if board[row][i]:
            return False

    # check upper left diagonal
    for i, j in zip(range(row, -1, -1),
                    range(col, -1, -1)):
        if board[i][j]:
            return False
    # check down left diagonal
    for i, j in zip(range(row, len(board)), range(col, -1, -1)):
        if board[i][j]:
            return False

    return True


# -----------------------------------------------------------------------
"""
8.13 Stack of Boxes: You have a stack of n boxes, with widths Wi' heights hi' and depths d1• The boxes
cannot be rotated and can only be stacked on top of one another if each box in the stack is strictly
larger than the box above it in width, height. and depth. Implement a method to compute the
height of the tallest possible stack. The height of a stack is the sum of the heights of each box.
"""


def max_height(stack_of_boxes):
    return recur_max_height(stack_of_boxes, 0, None, {})

    # time O(n^2)
    # space O(n)


def recur_max_height(stack, idx, curr_box, memo):
    if idx >= len(stack):
        return 0

    new_box = stack[idx]
    height_with_new_box = 0

    if not curr_box or can_above(curr_box, new_box):
        if idx not in memo:
            memo[idx] = recur_max_height(stack, idx + 1, new_box, memo)
            memo[idx] += new_box.height
        height_with_new_box = memo[idx]

    height_without_new_box = recur_max_height(stack, idx + 1, curr_box, memo)

    return max(height_with_new_box, height_without_new_box)


def can_above(prev_box, new_box):
    return prev_box.height > new_box.height and prev_box.width > new_box.width \
           and prev_box.depth > new_box.depth
