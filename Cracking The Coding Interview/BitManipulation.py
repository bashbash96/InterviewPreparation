def numOfDigits(num):
    count = 0
    while num > 0:
        num //= 10
        count += 1
    return count


def numOfBits(num):
    count = 0
    while num > 0:
        num >>= 1
        count += 1

    return count


def getBit(num, i):
    """
    get ith bit from num
    :param num:
    :param i: 0 <= i < bits(nums)
    :return:
    """
    if i < 0 or i >= numOfBits(num):
        return 0

    mask = (1 << i)
    currBit = num & mask
    return int(currBit != 0)


def setBit(num, i):
    """
    set ith bit in num
    :param num:
    :param i: 0 <= i < bits(nums)
    :return:
    """
    if i < 0 or i >= numOfBits(num):
        return num
    mask = 1 << i
    return num | mask


def clearBit(num, i):
    """
    clear ith bit in num
    :param num:
    :param i: 0 <= i < bits(nums)
    :return:
    """
    if i < 0 or i >= numOfBits(num):
        return num

    mask = (1 << i)
    mask = ~mask

    return num & mask


def clearAllLeft(num, i):
    """
    clear all bits from the MSB till i (contains i)
    :param num:  0 <= i < bits(nums)
    :param i:
    :return:
    """
    if i < 0 or i >= numOfBits(num):
        return num

    mask = (1 << i)
    mask -= 1
    return num & mask


def clearAllRight(num, i):
    """
    clear all bits from i (contains) till the LSB
    :param num:
    :param i: 0 <= i < bits(nums)
    :return:
    """
    if i < 0 or i >= numOfBits(num):
        return num

    # mask = (1 << (i+1))
    # mask -= 1
    # mask = ~ mask
    mask = (-1 << (i + 1))

    return num & mask


def updateBit(num, i, bit):
    """
    update the ith but with given bit
    :param num:
    :param i: 0 <= i < bits(nums)
    :param bit: 0/1 bit
    :return:
    """
    if bit != 1 and bit != 0:
        return "Error - the bit must be 0 or 1"

    num = clearBit(num, i)

    # mask = ~(1 << i)
    # num &= mask

    mask = (bit << i)
    return num | mask


# -----------------------------------------------------------------------
"""
5.1 Insertion: You are given two 32-bit numbers, Nand M, and two bit positions, i and j. Write a method
to insert Minto N such that M starts at bit j and ends at bit i. You can assume that the bits j through
i have enough space to fit all of M. That is, if M = 18811, you can assume that there are at least 5
bits between j and i. You would not, for example, have j = 3 and i = 2, because M could not fully
fit between bit 3 and bit 2.
EXAMPLE
Input: N Ul811 , i 2, j 6
Output: N = 18881881188
"""


def insert(N, M, i, j):
    allOnes = ~0

    left = allOnes << (j + 1)

    right = ~(allOnes << i)

    mask = left | right

    M = (M << i)
    N = (N & mask)

    return N | M

    # time O(bits(M) + bits(N))
    # space O(1)


# -----------------------------------------------------------------------
"""
5.2 Binary to String: Given a real number between 8 and 1 (e.g., 0.72) that is passed in as a double,
print the binary representation. If the number cannot be represented accurately in binary with at
most 32 characters, print "ERROR:'
"""


def binaryToString(num):
    if num > 1 or num <= 0:
        return "Error - num must be positive real"

    res = ['.']
    while num > 0:
        if len(res) > 32:
            return "Error - num can't be represented accurately"
        num *= 2
        if num >= 1:
            res.append('1')
            num -= 1
        else:
            res.append('0')

    return ''.join(res)

    # time O(1)


# -----------------------------------------------------------------------
"""
5.3 Flip Bit to Win: You have an integer and you can flip exactly one bit from a 0 to a 1. Write code to
find the length of the longest sequence of 1 s you could create.
EXAMPLE
Input: 1775
Output: 8
"""


def flipBit(num):
    if ~num == 0:
        return num

    prev, curr, maxLen = 0, 0, 0
    while num > 0:

        if num & 1 != 0:
            curr += 1
        else:
            if (num >> 1) & 1 == 0:
                prev = 0
            else:
                prev = curr

            curr = 0

        num >>= 1
        maxLen = max(maxLen, prev + curr + 1)

    return maxLen

    # time O(b)
    # space O(1)


# -----------------------------------------------------------------------
"""
5.4 Next Number: Given a positive integer, print the next smallest and the next largest number that
have the same number of 1 bits in their binary representation.
"""


def nextNumber(num):
    print(getSmallestBigger(num), getBiggestSmaller(num))


def getSmallestBigger(num):
    """
    found the first zero bit with at least one one bit to right of it suppose it found in
    place p, flip it to one after that put numOfOnes - 1 ones next to p and the
    rest is zeros
    :param num:
    :return:
    """
    foundOne = False
    copy = num
    place = 0
    numOfOnes = 0
    while copy > 0:
        if copy & 1 != 0:
            foundOne = True
            numOfOnes += 1
        else:
            if foundOne:
                break

        place += 1
        copy >>= 1

    num = (num | (1 << place))  # flip rightmost non trailing zero

    mask = (1 << place)
    mask -= 1  # all zeros followed by place ones
    mask = ~mask  # all ones followed by place zeros

    num &= mask  # clear all bits to the right of place

    mask2 = (1 << (numOfOnes - 1))
    mask2 -= 1

    return num | mask2  # insert (numOfOnes - 1) ones on the right of place


def getBiggestSmaller(num):
    """
    found the first one bit with at least one zero bit to right of it suppose
    it found in place p, flip it to zero after that put numOfZeros - 1 zeros at
    the most right, then the rest is ones
    :param num:
    :return:
    """
    copy = num

    numOfZeros = 0
    place = 0
    foundZero = False

    while copy > 0:
        if copy & 1 != 0:
            if foundZero:
                break
        else:
            foundZero = True
            numOfZeros += 1

        copy >>= 1
        place += 1

    num &= ~(1 << place)  # clear bit on place to zero

    mostRightZeros = ~0 - ((1 << (numOfZeros - 1)) - 1)
    num &= mostRightZeros  # insert numOfZeros - 1 zeroes on the most right

    pZeros = ~0 - ((1 << place) - 1)
    num |= mostRightZeros - pZeros  # insert the rest as ones

    return num


# -----------------------------------------------------------------------
"""
5.5 Debugger: Explain what the following code does: «n & (n-1)) == 0).
"""
# answer : check if n is power of two or n is zero

# -----------------------------------------------------------------------
"""
5.6 Conversion: Write a function to determine the number of bits you would need to flip to convert
integer A to integer B.
EXAMPLE
Input: 29 (or: 111(1), 15 (or: (1111)
Output: 2
"""


def conversion(num1, num2):
    xor = num1 ^ num2
    return countNumOfOnes(xor)


def countNumOfOnes(num):
    count = 0
    while num > 0:
        if num & 1 != 0:
            count += 1
        num >>= 1

    return count


# -----------------------------------------------------------------------
"""
5.7 Pairwise Swap: Write a program to swap odd and even bits in an integer with as few instructions as
possible (e.g., bit 0 and bit 1 are swapped, bit 2 and bit 3 are swapped, and so on).
"""


def pairWiseSwap(num):
    """
    take all the odd bits then shift them one to the right, take all even bits
    then shift them one to the left, finally merge them
    :param num:
    :return:
    """

    return ((num & 0xaaaaaaaa) >> 1) | ((num & 0x55555555) << 1)
