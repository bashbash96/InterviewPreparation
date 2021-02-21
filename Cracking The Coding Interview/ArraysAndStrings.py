# -----------------------------------------------------------------------
"""
1.1 Is Unique: Implement an algorithm to determine if a string has all unique characters. What if you
cannot use additional data structures?
"""


def isUnique(S):
    if len(S) > 257:
        return False

    unique = set()

    for char in S:
        if char in unique:
            return False

        unique.add(char)

    return True

    # time O(n)
    # space O(1) because there is 256 chars in ascii representation and the
    # loop won't iterate more than this number


# -----------------------------------------------------------------------
"""
1.2 Check Permutation: Given two strings, write a method to decide if one is a permutation of the
other.
"""


def checkPermutation(a, b):
    if len(a) != len(b):
        return False

    count = {}

    for char in a:
        if char in count:
            count[char] += 1
        else:
            count[char] = 1

    for char in b:
        if char not in count:
            return False
        else:
            count[char] -= 1
            if count[char] < 0:
                return False

    return True

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
1.3 URLify: Write a method to replace all spaces in a string with '%20: You may assume that the string
has sufficient space at the end to hold the additional characters, and that you are given the "true"
length of the string. (Note: If implementing in Java, please use a character array so that you can
perform this operation in place.)
"""


def URLify(s):
    lst = [''] * len(s)
    j = 0
    i = 0
    while i < len(s) and j + 2 < len(lst):

        if s[i] == ' ':
            lst[j] = '%'
            lst[j + 1] = '2'
            lst[j + 2] = '0'
            j += 3
            i += 1
        else:
            lst[j] = s[i]
            i += 1
            j += 1

    return ''.join(lst)

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
1.4 Palindrome Permutation: Given a string, write a function to check if it is a permutation of a palindrome.
A palindrome is a word or phrase that is the same forwards and backwards. A permutation
is a rea rearrangement of letters. The palindrome does not need to be limited to just dictionary words.
"""


def countChars(s):
    count = {}
    for char in s:
        if char in count:
            count[char] += 1
        else:
            count[char] = 1

    return count


def palindromePerm(s):
    s = s.lower()
    s = s.replace(' ', '')

    count = countChars(s)

    foundOdd = False
    for key in count:
        if count[key] % 2 != 0:
            if foundOdd:
                return False

            foundOdd = True

    return True

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
1.5 One Away: There are three types of edits that can be performed on strings: insert a character,
remove a character, or replace a character. Given two strings, write a function to check if they are
one edit (or zero edits) away.
"""


def oneReplace(s1, s2):
    found = False
    for i in range(len(s1)):

        if s1[i] != s2[i]:
            if found:
                return False
            found = True

    return True


def oneInsert(s1, s2):
    idx1, idx2 = 0, 0

    while idx1 < len(s1) and idx2 < len(s2):

        if s1[idx1] != s2[idx2]:

            if idx1 != idx2:
                return False
            idx2 += 1
        else:
            idx1 += 1
            idx2 += 1

    return True


def oneAway(s1, s2):
    if len(s1) == len(s2):
        return oneReplace(s1, s2)
    elif len(s1) + 1 == len(s2):
        return oneInsert(s1, s2)
    elif len(s2) + 1 == len(s1):
        return oneInsert(s2, s1)

    return False

    # time O(n) -- n the length of the shorter string
    # space O(1)


# -----------------------------------------------------------------------
"""
1.6 String Compression: Implement a method to perform basic string compression using the counts
of repeated characters. For example, the string aabcccccaaa would become a2b1c5a3. If the
"compressed" string would not become smaller than the original string, your method should return
the original string. You can assume the string has only uppercase and lowercase letters (a - z).
"""


def countOccurrences(s, currChar, i):
    currCharCount = 0

    while i < len(s) and s[i] == currChar:
        currCharCount += 1
        i += 1

    return currCharCount


def compress(s):
    res = []

    i = 0

    while i < len(s):
        currChar = s[i]
        currCharCount = countOccurrences(s, currChar, i)

        res.append(currChar)
        res.append(str(currCharCount))
        if len(res) >= len(s):
            return s

        i += currCharCount

    if len(res) < len(s):
        return ''.join(res)

    return s

    # time O(p + k) where p is the length of the string and k is the length
    # of the result string
    # space O(k)


# -----------------------------------------------------------------------
"""
1.7 Rotate Matrix: Given an image represented by an NxN matrix, where each pixel in the image is 4
bytes, write a method to rotate the image by 90 degrees. (an you do this in place?
"""


def rotate(arr, row, col):
    topleft = arr[row][col]
    arr[row][col] = arr[len(arr) - col - 1][row]  # bottom left to top right
    arr[len(arr) - col - 1][row] = arr[len(arr) - row - 1][len(arr) - col - 1]  # bottom right to bottom left
    arr[len(arr) - row - 1][len(arr) - col - 1] = arr[col][len(arr) - row - 1]  # top right to bottom left
    arr[col][len(arr) - row - 1] = topleft  # top left to top right


def rotateMatrix(arr):
    for row in range(0, len(arr) // 2):
        for col in range(row, len(arr) - row - 1):
            rotate(arr, row, col)

    return arr

    # time O(n^2)
    # space O(1)


# -----------------------------------------------------------------------
"""
1.8 Zero Matrix: Write an algorithm such that if an element in an MxN matrix is 0, its entire row and
column are set to O.
"""


def checkForZeros(arr):
    rows = [False] * len(arr)
    cols = [False] * len(arr[0])

    for row in range(len(arr)):
        for col in range(len(arr[0])):
            if arr[row][col] == 0:
                rows[row] = True
                cols[col] = True

    return rows, cols


def makeZeros(arr, arr2, str_):
    if str_ == 'row':
        for row in range(len(arr2)):
            if arr2[row]:
                for col in range(len(arr[0])):
                    arr[row][col] = 0
    else:
        for col in range(len(arr2)):
            if arr2[col]:
                for row in range(len(arr)):
                    arr[row][col] = 0


def makeZero(arr):
    rows, cols = checkForZeros(arr)

    makeZeros(arr, rows, 'row')
    makeZeros(arr, cols, 'col')

    return arr

    # time O(n * m)
    # space O(n + m)


# -----------------------------------------------------------------------
"""
1.9 String Rotation: Assume you have a method isSubst ring which checks if one word is a substring
of another. Given two strings, 51 and 52, write code to check if 52 is a rotation of 51 using only one
call to isSubstring (e.g., "waterbottle" is a rotation of"erbottlewat").
"""


def isRotationWithOneCall(s1, s2):
    if len(s2) != len(s1):
        return False

    s1 = s1 + s1

    return s2 in s1

    # time O(m + n)
    # space O(len(s1))
