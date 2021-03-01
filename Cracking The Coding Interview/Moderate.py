import sys
from collections import defaultdict

# -----------------------------------------------------------------------
"""
16.1 Number Swapper: Write a function to swap a number in place (that is, without temporary variables).
Hints: #492, #716, #737
"""


def numSwapper(num1, num2):
    # with maths
    # num1 = num1 + num2
    # num2 = num1 - num2
    # num1 = num1 - num2

    # with bit manipulation
    num1 = num1 ^ num2
    num2 = num1 ^ num2
    num1 = num1 ^ num2

    return num1, num2

    # time O(x) -> num of the bits
    # space O(1)


# -----------------------------------------------------------------------
"""
16.2 Word Frequencies: Design a method to find the frequency of occurrences of any given word in a
book. What if we were running this algorithm multiple times?
"""


def pre_process(book):
    table = defaultdict(int)

    for word in book:
        table[word.lower()] += 1

    return table

    # time O(n)
    # space O(n)


def get_frequency(table, word):
    return table.get(word, 0)

    # time O(1)
    # space O(1)


# -----------------------------------------------------------------------
"""
16.5 Factorial Zeros: Write an algorithm which computes the number of trailing zeros in n factorial.
"""


def factZeros(num):
    trailingZeros = 0

    if num < 5:
        return trailingZeros
    for num in range(5, num + 1, 5):
        trailingZeros += numOfFive(num)

    return trailingZeros

    # time O(n)
    # space O(1)


def numOfFive(num):
    count = 0
    while num > 0:
        if num % 5 == 0:
            count += 1
        num //= 5

    return count


# -----------------------------------------------------------------------
"""
16.6 Smallest Difference: Given two arrays of integers, compute the pair of values (one value in each
array) with the smallest (non-negative) difference. Return the difference.
EXAMPLE
Input{1,3,15,11,2}, {23, 127, 235,19,8}
Output 3. That is, the pair (11, 8).
"""


def smallestDiff(arr1, arr2):
    if not arr1 or not arr2:
        return float('inf')

    arr1.sort()
    arr2.sort()

    p1, p2 = 0, 0
    minDiff = float('inf')
    while p1 < len(arr1) and p2 < len(arr2):
        currDiff = abs(arr1[p1] - arr2[p2])
        if currDiff < minDiff:
            minDiff = currDiff

        if arr1[p1] < arr2[p2]:
            p1 += 1
        else:
            p2 += 1

    return minDiff

    # time O(m*log(m) + n*log(n))
    # space O(1)


# -----------------------------------------------------------------------
"""
16.7 Number Max: Write a method that finds the maximum of two numbers. You should not use if-else
or any other comparison operator.
"""


def flip(bit):
    # flip bit to opposite bit
    return 1 ^ bit


def sign(num):
    # return 1 if the num is positive else 0
    return flip((num >> 31) & 0x1)


def maxNum(num1, num2):
    diff = num1 - num2

    s1 = sign(num1)
    s2 = sign(num2)
    sd = sign(diff)

    sign1 = s1 ^ s2
    signd = flip(sign1)

    k = sign1 * s1 + signd * sd

    q = flip(k)

    return num1 * k + num2 * q


# -----------------------------------------------------------------------
"""
16.8 English int: Given any integer, print an English phrase that describes the integer (e.g ., "One
Thousand, Two Hundred Thirty Four").
"""

smalls = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
          "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
bigs = ["", "Thousand", "Million", "Billion"]
negative = "Negative"
hundred = 'Hundred'


def convert(n):
    if n == 0:
        return smalls[0]
    elif n < 0:
        return negative + ' ' + convert(-1 * n)
    res = []
    count = 0
    while n > 0:
        if n % 1000 != 0:
            if count == 0:
                res.insert(0, convertChunk(n % 1000))
            else:
                res.insert(0, convertChunk(n % 1000) + ' ' + bigs[count] + ',')
        n //= 1000
        count += 1

    return ' '.join(res)


def convertChunk(n):
    res = []
    if n >= 100:
        res.append(smalls[n // 100] + ' ' + hundred)
        n %= 100

    if n >= 20:
        res.append(tens[(n // 10)])
        n %= 10

    if 0 < n < 20:
        res.append(smalls[n])

    return ' '.join(res)


# -----------------------------------------------------------------------
"""
16.9 Operations: Write methods to implement the multiply, subtract, and divide operations for integers.
The results of all of these are integers. Use only the add operator.
"""


def operations(n, m, op):
    if op == '-':
        return sub(n, m)
    elif op == '*':
        if n > 0 and m > 0:
            return mult(n, m)
        elif n < 0 and m < 0:
            return mult(changeSign(m), changeSign(n))
        else:
            return changeSign(mult(changeSign(min(m, n)), max(m, n)))
    else:
        if n > 0 and m > 0:
            return div(n, m)
        elif n < 0 and m < 0:
            return div(changeSign(n), changeSign(m))
        else:
            if n < 0:
                n = changeSign(n)
            if m < 0:
                m = changeSign(m)
            division = div(n, m)
            if division == 'Error':
                return division
            return changeSign(division)


def mult(n, m):
    if n < m:
        return mult(m, n)
    res = 0
    while m > 0:
        res += n
        m = sub(m, 1)

    return res


def changeSign(n):
    negative = 0
    sign = 1 if n < 0 else -1

    while n != 0:
        negative += sign
        n += sign

    return negative


def sub(n, m):
    return n + changeSign(m)


def div(n, m):
    res = 0
    if m == 0:
        return 'Error'
    while n > m:
        res += 1
        n = sub(n, m)

    return res


# -----------------------------------------------------------------------
"""
16.10 Living People: Given a list of people with their birth and death years, implement a method to
compute the year with the most number of people alive. You may assume that all people were born
between 1900 and 2000 (inclusive). If a person was alive during any portion of that year, they should
be included in that year's count. For example, Person (birth = 1908, death = 1909) is included in the
counts for both 1908 and 1909.
"""


def get_max_alive_year(persons):
    res = []
    for p in persons:
        res.append(p.birthYear(), 'b')
        res.append(p.deathYear(), 'd')

    res = sorted(res, key=lambda pair: pair[0])
    count = 0
    maxVal = 0
    maxYear = 0
    for pair in res:
        if pair[1] == 'b':
            count += 1
            if count > maxVal:
                maxVal = count
                maxYear = pair[0]
        else:
            count -= 1

    return maxYear


# -----------------------------------------------------------------------
"""
16.15 Master Mind: The Game of Master Mind is played as follows:
The computer has four slots, and each slot will contain a ball that is red (R), yellow (Y), green (G) or
blue (B). For example, the computer might have RGGB (Slot #1 is red, Slots #2 and #3 are green, Slot
#4 is blue).
You, the user, are trying to guess the solution. You might, for example, guess YRGB.
When you guess the correct color for the correct slot, you get a "hit:' If you guess a color that exists
but is in the wrong slot, you get a "pseudo-hit:' Note that a slot that is a hit can never count as a
pseudo-hit.
For example, if the actual solution is RGBY and you guess GGRR, you have one hit and one pseudohit.
Write a method that, given a guess and a solution, returns the number of hits and pseudo-hits.
"""


def countHitsAndPseudoHits(solution, guess):
    if len(solution) != len(guess):
        return 'ERROR'

    sCount = countFreq(solution)
    gCount = countFreq(guess)

    hits = countHits(solution, guess, sCount, gCount)
    pseudoHits = countPseudoHits(sCount, gCount)

    print(hits, pseudoHits)

    # time O(n)
    # space O(n)


def countFreq(st):
    res = {}
    for c in st:
        if c in res:
            res[c] += 1
        else:
            res[c] = 1

    return res


def countHits(solution, guess, sCount, gCount):
    counter = 0
    for i in range(len(solution)):
        if solution[i] == guess[i]:
            counter += 1
            sCount[solution[i]] -= 1
            gCount[guess[i]] -= 1

    return counter


def countPseudoHits(sCount, gCount):
    counter = 0
    for c in sCount:
        if sCount[c] != 0 and c in gCount and gCount[c] != 0:
            counter += min(sCount[c], gCount[c])

    return counter


# -----------------------------------------------------------------------
"""
16.16 Sub Sort: Given an array of integers, write a method to find indices m and n such that if you sorted
elements m through n, the entire array would be sorted. Minimize n - m (that is, find the smallest
such sequence).
EXAMPLE
Input: 1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19
Output: (3, 9)
"""


def subSort(arr):
    leftIdx = getLeftSub(arr)
    rightIdx = getRightSub(arr)

    minIdx = leftIdx
    maxIdx = rightIdx

    for i in range(leftIdx + 1, rightIdx):
        if arr[i] < arr[minIdx]:
            minIdx = i
        if arr[i] > arr[maxIdx]:
            maxIdx = i

    leftIdx = shrinkLeft(arr, minIdx, leftIdx)
    rightIdx = shrinkRight(arr, maxIdx, rightIdx)

    return leftIdx, rightIdx


def shrinkLeft(arr, minIdx, leftIdx):
    for i in range(leftIdx, -1, -1):
        if arr[i] <= arr[minIdx]:
            return i + 1
    return 0


def shrinkRight(arr, maxIdx, rightIdx):
    for i in range(rightIdx, len(arr)):
        if arr[i] >= arr[maxIdx]:
            return i - 1
    return len(arr) - 1


def getLeftSub(arr):
    left = 0

    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            return left
        else:
            left = i

    return left


def getRightSub(arr):
    right = len(arr) - 1

    for i in range(len(arr) - 2, -1, -1):
        if arr[i] > arr[i + 1]:
            return right
        else:
            right = i

    return right


# -----------------------------------------------------------------------
"""
16.17 Contiguous Sequence: You are given an array of integers (both positive and negative). Find the
contiguous sequence with the largest sum. Return the sum.
EXAMPLE
Input 2, -8, 3, -2, 4, -10
OutputS (i.e., {3, -2, 4})
"""


def contiguousSeq(arr):
    memo = [val for val in arr]

    for i in range(1, len(arr)):
        if memo[i] < memo[i - 1] + arr[i]:
            memo[i] = memo[i - 1] + arr[i]

    return max(memo)

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
16.19 Pond Sizes: You have an integer matrix representing a plot of land, where the value at that location
represents the height above sea level. A value of zero indicates water. A pond is a region of water
connected vertically, horizontally, or diagonally. The size of the pond is the total number of
connected water cells. Write a method to compute the sizes of all ponds in the matrix.
EXAMPLE
Input:
0 2 1 0
0 1 0 1
1 1 0 1
0 1 0 1
Output: 2, 4, 1 (in any order)
"""


def pondSizes(mat):
    if len(mat) == 0:
        return -1

    res = []
    visited = [[False for j in range(len(mat[0]))] for i in range(len(mat))]

    for row in range(len(mat)):
        for col in range(len(mat[0])):
            if not visited[row][col] and mat[row][col] == 0:
                res.append(calculateSize(mat, row, col, visited))
    return res

    # time O(n*m)
    # space O(n*m)


def calculateSize(mat, row, col, visited):
    if row < 0 or col < 0 or row >= len(mat) or col >= len(mat[0]) or visited[row][col] or mat[row][col] != 0:
        return 0
    visited[row][col] = True

    size = 1
    for nextRow in range(-1, 2):
        for nextCol in range(-1, 2):
            size += calculateSize(mat, row + nextRow, col + nextCol, visited)

    return size

    # up = calculateSize(mat, row - 1, col, visited)
    # down = calculateSize(mat, row + 1, col, visited)
    # left = calculateSize(mat, row, col - 1, visited)
    # right = calculateSize(mat, row, col + 1, visited)
    # leftUp = calculateSize(mat, row - 1, col - 1, visited)
    # leftDown = calculateSize(mat, row + 1, col - 1, visited)
    # rightUp = calculateSize(mat, row - 1, col + 1, visited)
    # rightDown = calculateSize(mat, row + 1, col + 1, visited)
    # curr = 0
    # if mat[row][col] == 0:
    #     curr += 1
    # return up + down + left + right + leftUp + leftDown + rightUp + rightDown + curr


# -----------------------------------------------------------------------
"""
16.20 T9: On old cell phones, users typed on a numeric keypad and the phone would provide a list of words
that matched these numbers. Each digit mapped to a set of 0 - 4 letters. Implement an algorithm
to return a list of matching words, given a sequence of digits. You are provided a list of valid words
(provided in whatever data structure you'd like). The mapping is shown in the diagram below:
EXAMPLE
Input:
Output:
SOLUTION
8733
tree, used
"""

"""
Solution:

mapping = {1: [], 2: ['a', 'b', 'c'], 3: ['d', 'e', 'f'], 4: ['g', 'h', 'i'], 5: ['j', 'k', 'l'],
           6: ['m', 'n', 'o'], 7: ['p', 'q', 'r', 's'], 8: ['t', 'u', 'v'], 9: ['w', 'x', 'y', 'z']}


def validWords(num):
    trie = Trie() -> # Trie tree with the valid words, it has starts with to check if word start with
                        # a certain prefix, and search for specific word..
    
    list = getListFormNum(num)
    res = []
    digit = list[0]
    for char in mapping[digit]:
        if trie.startsWith(char):
            checkValid(res, list, trie, 1, char)
    
    # time O(4^log(n)^2)

def checkValid(res, list, trie, idx, currWord):
    if idx == len(list):
        if trie.search(currWord)
            res.append(currWord)
        return

    for char in mapping[list[idx]]:
        if trie.startsWith(currWord + char):
            checkValid(res, list, trie, idx+1, currWord + char)
            
    
def getListFromNum(num):
    res = []
    while num > 0:
        res.append(num%10)
        num //= 10
    res.reverse()
    return res
"""

# -----------------------------------------------------------------------
"""
16.21 Sum Swap: Given two arrays of integers, find a pair of values (one value from each array) that you
can swap to give the two arrays the same sum.
EXAMPLE
Input:{4, l, 2, l, l, 2} and {3, 6, 3, 3}
Output: {l, 3}
"""


def sumSwap(arr1, arr2):
    setNums = set()

    for num in arr1:
        setNums.add(num)

    sum1 = sum(arr1)
    sum2 = sum(arr2)

    if (sum1 + sum2) % 2 != 0:
        return 'ERROR'

    mid = (sum1 + sum2) // 2

    for num in arr2:
        currSum2 = sum2 - num
        comp = mid - currSum2
        if comp in setNums:
            return num, comp

    return -1

    # time O(n+m)
    # space O(n)


# -----------------------------------------------------------------------
"""
16.23 Rand7 from RandS: Implement a method rand7() given randS(). That is, given a method that
generates a random number between 0 and 4 (inclusive), write a method that generates a random
number between 0 and 6 (inclusive).
"""

"""
def rand7():
    while True:
        num = 5 * rand5() + rand5() # -> range from (0,24) and the range we want is (0,20) to make sure every number
                                    # will be chosen three times...
        if num < 21:
            return num % 7
"""

# -----------------------------------------------------------------------
"""
16.24 Pairs with Sum: Design an algorithm to find all pairs of integers within an array which sum to a
specified value.
"""


def pairsWithSum(arr, sum):
    setNums = set()
    for num in arr:
        setNums.add(num)

    res = []
    for num in arr:
        if sum - num in setNums:
            setNums.remove(num)
            res.append((num, sum - num))

    return res


# -----------------------------------------------------------------------
"""
16.26 Calculator: Given an arithmetic equation consisting of positive integers, +, -, * and / (no parentheses),
compute the result.
EXAMPLE
Input: 2*3+5/6*3+15
Output: 23.5
"""


def calc(equation):
    stack = []

    operations = {'*', '-', '+', '/'}

    i = 0
    while i < len(equation):
        if not isValid(equation[i]):
            return 'ERROR - invalid equation'

        if equation[i] in operations:
            if len(stack) == 0 or stack[len(stack) - 1] in operations:
                return 'ERROR - invalid equation'

            if equation[i] == '*' or equation[i] == '/':
                first = getNumber(stack)
                op = equation[i]
                if i + 1 >= len(equation) or equation[i + 1] in operations:
                    return 'ERROR - invalid equation'
                second, idx = getNumFromEqu(equation, i + 1)
                res = makeOp(first, second, op)
                stack.append(res)
                i = idx - 1
            else:
                stack.append(equation[i])
        else:
            stack.append(float(equation[i]))
        i += 1

    while len(stack) > 1:
        first = getNumber(stack)
        op = stack.pop()
        second = getNumber(stack)
        res = makeOp(second, first, op)
        stack.append(res)

    return stack[0]


def isValid(c):
    valid = []
    for i in range(10):
        valid.append(str(i))

    valid.extend(['+', '-', '*', '/'])

    if c in valid:
        return True

    return False


def getNumFromEqu(equation, i):
    res = 0
    operations = {'*', '-', '+', '/'}
    while i < len(equation) and equation[i] not in operations:
        res = res * 10 + float(equation[i])
        i += 1

    return res, i


def makeOp(first, second, op):
    if op == '+':
        return first + second
    if op == '-':
        return first - second
    if op == '*':
        return first * second
    if op == '/':
        return first / second


def getNumber(stack):
    counter = 0
    operations = {'*', '-', '+', '/'}

    res = 0

    while len(stack) > 0:
        curr = stack.pop()
        if curr in operations:
            stack.append(curr)
            return res
        res = res + float(curr) * pow(10, counter)

        counter += 1

    return res
