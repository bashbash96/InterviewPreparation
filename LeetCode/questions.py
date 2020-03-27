# -----------------------------------------------------------------------
"""
122. Best Time to Buy and Sell Stock II
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:

Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Example 2:

Input: [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
             Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
             engaging multiple transactions at the same time. You must sell before buying again.
Example 3:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
"""


class Solution:
    def maxProfit(self, prices):
        if len(prices) < 1:
            return 0
        low, high = prices[0], prices[0]
        profit = 0
        i = 0
        while i < len(prices) - 1:
            while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
                i += 1
            low = prices[i]

            while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
                i += 1
            high = prices[i]

            profit += high - low

        return profit

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
350. Intersection of Two Arrays II

Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Note:

Each element in the result should appear as many times as it shows in both arrays.
The result can be in any order.

Follow up:
What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to nums2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?
"""


class Solution:

    # def intersect(self, nums1, nums2):
    #
    #     if len(nums1) == 0 or len(nums2) == 0:
    #         return []
    #
    #     nums1.sort()
    #     nums2.sort()
    #     res = []
    #
    #     p1, p2 = 0, 0
    #     while p1 < len(nums1) and p2 < len(nums2):
    #         if nums1[p1] < nums2[p2]:
    #             p1 += 1
    #         elif nums1[p1] > nums2[p2]:
    #             p2 += 1
    #         else:
    #             res.append(nums1[p1])
    #             p1 += 1
    #             p2 += 1
    #
    #     return res
    #
    #     # time O(n * log(n) + m * log(m))
    #     # space O(min(n,m))
    #

    def intersect(self, nums1, nums2):
        if len(nums1) == 0 or len(nums2) == 0:
            return []

        shorter = nums1 if len(nums1) < len(nums2) else nums2
        longer = nums2 if len(nums2) > len(nums1) else nums1

        numsCount = {}
        for num in shorter:
            if num in numsCount:
                numsCount[num] += 1
            else:
                numsCount[num] = 1

        res = []
        for num in longer:
            if num in numsCount and numsCount[num] > 0:
                res.append(num)
                numsCount[num] -= 1

        return res

        # time O(n + m)
        # space O(min(n,m))


# -----------------------------------------------------------------------
"""
118. Pascal's Triangle

Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.


In Pascal's triangle, each number is the sum of the two numbers directly above it.

Example:

Input: 5
Output:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
"""


class Solution:
    def generate(self, numRows):

        if numRows <= 0:
            return []
        elif numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1, 1]]

        res = [[1], [1, 1]]

        for i in range(2, numRows):
            currRes = [1]
            j = 1
            prevIdx = i - 1
            prevList = res[prevIdx]
            while j < len(prevList):
                currNum = prevList[j] + prevList[j - 1]
                currRes.append(currNum)
                j += 1
            currRes.append(1)
            res.append(currRes)

        return res

        # time O(n^2)
        # space O(n^2)


# -----------------------------------------------------------------------
"""
191. Number of 1 Bits
Write a function that takes an unsigned integer and return the number of '1' bits it has (also known as the Hamming weight).

Example 1:

Input: 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.

Example 2:

Input: 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.

"""


class Solution:
    def hammingWeight(self, n: int) -> int:

        count = 0

        while n > 0:
            if n & 1 != 0:
                count += 1
            n >>= 1

        return count

        # time O(log(n))
        # space O(1)


# -----------------------------------------------------------------------
"""
101. Symmetric Tree

Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
 

But the following [1,2,2,null,3,null,3] is not:

    1
   / \
  2   2
   \   \
   3    3
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # def isSymmetric(self, root):
    #
    #     if not root:
    #         return True
    #
    #     q = [root]
    #
    #     while len(q) > 0:
    #
    #         if not self.isSym(q):
    #             return False
    #
    #         levelLen = len(q)
    #
    #         while levelLen > 0:
    #             curr = q.pop(0)
    #             if curr:
    #                 q.append(curr.left)
    #                 q.append(curr.right)
    #
    #             levelLen -= 1
    #
    #     return True
    #
    # def isSym(self, level):
    #     start, end = 0, len(level) - 1
    #
    #     while start < end:
    #         left = level[start]
    #         right = level[end]
    #
    #         if left and right:
    #             if left.val != right.val:
    #                 return False
    #         elif (left and not right) or (not left and right):
    #             return False
    #
    #         start += 1
    #         end -= 1
    #
    #     return True
    #
    #     # time O(n)
    #     # space (O(maximum level length))
    #
    def isSymmetric(self, root):

        return self.recurIsSym(root, root)

        # time O(n)
        # space O(n)

    def recurIsSym(self, root1, root2):
        if not root1 and not root2:
            return True

        if not root1 or not root2:
            return False

        return root1.val == root2.val and (
                self.recurIsSym(root1.left, root2.right) and self.recurIsSym(root1.right, root2.left))


# -----------------------------------------------------------------------
"""
66. Plus One

Given a non-empty array of digits representing a non-negative integer, plus one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

Example 1:

Input: [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.

"""


class Solution:
    def plusOne(self, digits):

        digits.reverse()
        Sum = digits[0] + 1
        carry = Sum // 10
        Sum = Sum % 10
        digits[0] = Sum

        i = 1
        while i < len(digits):
            Sum = digits[i] + carry
            carry = Sum // 10
            Sum = Sum % 10
            digits[i] = Sum
            i += 1

        if carry != 0:
            digits.append(carry)

        return reversed(digits)

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
155. Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.

"""


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.min = None


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.head = None

    def push(self, x):
        node = Node(x)
        if not self.head:
            self.head = node
            node.min = x
            return

        node = Node(x)
        if self.head.min < x:
            node.min = self.head.min
        else:
            node.min = x

        node.next = self.head
        self.head = node

        # time O(1)

    def pop(self):
        if not self.head:
            return None
        data = self.head.data
        self.head = self.head.next
        return data

        # time O(1)

    def top(self):
        if not self.head:
            return None

        return self.head.data

        # time O(1)

    def getMin(self):
        if not self.head:
            return None
        return self.head.min

        # time O(1)


# -----------------------------------------------------------------------
"""
160. Intersection of Two Linked Lists

Write a program to find the node at which the intersection of two singly linked lists begins.

"""


class Solution:
    def getIntersectionNode(self, headA, headB):

        countA, countB = 0, 0
        curr = headA
        while curr:
            countA += 1
            curr = curr.next

        curr = headB
        while curr:
            countB += 1
            curr = curr.next

        if countA < countB:
            longer = headB
            shorter = headA
        else:
            longer = headA
            shorter = headB

        diff = abs(countA - countB)

        while diff > 0:
            longer = longer.next
            diff -= 1

        while longer and shorter and longer != shorter:
            longer = longer.next
            shorter = shorter.next

        return longer

        # time O(n + m)
        # space O(1)


# -----------------------------------------------------------------------
"""
20. Valid Parentheses

Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.
"""


class Solution:
    def isValid(self, s: str) -> bool:
        stack = []

        for char in s:
            if char == '(' or char == '[' or char == '{':
                stack.append(char)
            else:
                if char == ')':
                    if len(stack) == 0 or stack.pop() != '(':
                        return False
                elif char == ']':
                    if len(stack) == 0 or stack.pop() != '[':
                        return False
                else:
                    if len(stack) == 0 or stack.pop() != '{':
                        return False

        return len(stack) == 0

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
234. Palindrome Linked List

Given a singly linked list, determine if it is a palindrome.

Example 1:

Input: 1->2
Output: false
Example 2:

Input: 1->2->2->1
Output: true
Follow up:
Could you do it in O(n) time and O(1) space?

"""


class Solution:
    def isPalindrome(self, head):

        if not head or not head.next:
            return True

        slow, fast = head, head.next
        prev = head
        while slow and fast:
            fast = fast.next
            if fast:
                fast = fast.next
            prev = slow
            slow = slow.next

        prev.next = None

        reverseList = self.reverseList(slow)

        p1, p2 = head, reverseList

        while p1 and p2:
            if p1.val != p2.val:
                return False
            p1 = p1.next
            p2 = p2.next

        return True

        # time O(n)
        # space O(1)

    def reverseList(self, head):
        prev, curr, next = None, head, None

        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next

        return prev


# -----------------------------------------------------------------------
"""
172. Factorial Trailing Zeroes

Given an integer n, return the number of trailing zeroes in n!.

Example 1:

Input: 3
Output: 0
Explanation: 3! = 6, no trailing zero.
Example 2:

Input: 5
Output: 1
Explanation: 5! = 120, one trailing zero.
Note: Your solution should be in logarithmic time complexity.
"""


class Solution:
    def trailingZeroes(self, n):
        counter = 0

        while n >= 5:
            counter += n // 5
            n //= 5

        return counter


# -----------------------------------------------------------------------
"""
190. Reverse Bits

Reverse bits of a given 32 bits unsigned integer.

 

Example 1:

Input: 00000010100101000001111010011100
Output: 00111001011110000010100101000000
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, 
so return 964176192 which its binary representation is 00111001011110000010100101000000.

"""


class Solution:
    def reverseBits(self, n):
        res = []

        while n > 0:
            curr = n % 2
            res.append((curr))
            n //= 2

        diff = 32 - len(res)

        res = res + [0] * diff

        res.reverse()

        num = 0

        for i in range(len(res)):
            num = num + pow(2, i) * res[i]

        return num

        # time O(log(n))
        # space O(log(n))


# -----------------------------------------------------------------------
"""
14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.

"""


class Solution:
    def longestCommonPrefix(self, strs):
        if len(strs) == 0:
            return ''

        LCP = strs[0]

        for i in range(1, len(strs)):
            LCP = self.getCommon(LCP, strs[i])
            if LCP == '':
                return ''

        return LCP

        # time O(n)
        # space O(k) -> k : max length for common prefix

    def getCommon(self, str1, str2):
        res = []

        p1, p2 = 0, 0

        while p1 < len(str1) and p2 < len(str2) and str1[p1] == str2[p2]:
            res.append(str1[p1])
            p1 += 1
            p2 += 1

        return '' if len(res) == 0 else ''.join(res)


# -----------------------------------------------------------------------
"""
125. Valid Palindrome

Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

Note: For the purpose of this problem, we define empty string as valid palindrome.

Example 1:

Input: "A man, a plan, a canal: Panama"
Output: true
Example 2:

Input: "race a car"
Output: false

"""


class Solution:
    def isPalindrome(self, s):
        if len(s) == 0:
            return True

        alphaNum = []

        for c in s:
            if c.isalnum():
                if c.isalpha():
                    alphaNum.append(c.lower())
                else:
                    alphaNum.append(c)

        start, end = 0, len(alphaNum) - 1

        while start < end:
            if alphaNum[start] != alphaNum[end]:
                return False
            start += 1
            end -= 1

        return True

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
28. Implement strStr()

Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Example 1:

Input: haystack = "hello", needle = "ll"
Output: 2

"""


class Solution:
    def strStr(self, haystack, needle):
        subLen = len(needle)

        if len(needle) == 0:
            return 0
        if len(haystack) == 0:
            return -1

        for i in range(len(haystack)):
            if len(haystack) - i < subLen:
                break

            if haystack[i] == needle[0]:
                if self.isContain(haystack, i, needle):
                    return i

        return -1

        # time O(n * m)
        # space O(1)

    def isContain(self, str, i, sub):
        j = 0
        while i < len(str) and j < len(sub):
            if str[i] != sub[j]:
                return False

            i += 1
            j += 1

        return True


# -----------------------------------------------------------------------
"""
69. Sqrt(x)

Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example 1:

Input: 4
Output: 2
Example 2:

Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned
             
"""


class Solution:
    def mySqrt(self, x):
        root = 1

        while root * root <= x:
            root += 1

        return root - 1

        # time O(log(n))
        # space O(1)


# -----------------------------------------------------------------------
"""
204. Count Primes

Count the number of prime numbers less than a non-negative number, n.

Example:

Input: 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.
"""


class Solution:
    def countPrimes(self, n):
        if n < 2:
            return 0
        res = [True] * n

        res[0] = res[1] = False

        currPrime = 2

        while currPrime < n / 2:
            self.removeAllNums(res, currPrime)
            currPrime = self.nextPrime(res, currPrime)

        counter = 0
        for val in res:
            if val:
                counter += 1

        return counter

        # time O(n^2)
        # space O(n)

    def removeAllNums(self, res, prime):
        i = prime + prime
        while i < len(res):
            res[i] = False
            i += prime

    def nextPrime(self, res, prime):
        curr = prime + 1

        while not res[curr]:
            curr += 1

        return curr


# -----------------------------------------------------------------------
"""
7. Reverse Integer

Given a 32-bit signed integer, reverse digits of an integer.

Example 1:

Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
Note:
Assume we are dealing with an environment which could only store integers within
the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, 
assume that your function returns 0 when the reversed integer overflows.

"""


class Solution:
    def reverse(self, x):
        rev = 0
        neg = False

        if x < 0:
            x *= -1
            neg = True

        while x > 0:
            rev = rev * 10 + x % 10
            x //= 10

        if rev > pow(2, 31) - 1 or rev < pow(2, -31):
            rev = 0

        return rev if not neg else -1 * rev

        # time O(log(n))
        # space O(log(n))


# -----------------------------------------------------------------------
"""
46. Permutations

Given a collection of distinct integers, return all possible permutations.

Example:

Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

"""


class Solution:
    def permute(self, nums):
        if len(nums) == 0:
            return nums

        res = {}

        for i in range(len(nums)):

            if i - 1 in res:
                prevPerm = res[i - 1]
                currNum = nums[i]
                currPerm = []
                for prev in prevPerm:
                    for j in range(len(prev) + 1):
                        currPerm.append(prev[:j] + [currNum] + prev[j:])

                res[i] = currPerm
                del res[i - 1]
            else:
                res[i] = [[nums[i]]]

        return res[len(nums) - 1]

        # time O(2^n)
        # space O(2^n)


# -----------------------------------------------------------------------
"""
22. Generate Parentheses

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

"""


class Solution:
    def generateParenthesis(self, n):
        res = []
        self.generate(n, '', n, n, res)

        return res

    def generate(self, n, prefix, left, right, res):
        if left == 0 and right == 0:
            return res.append(prefix)

        if left > right or left < 0 or right < 0:
            return

        if left <= right:
            self.generate(n, prefix + '(', left - 1, right, res)
            self.generate(n, prefix + ')', left, right - 1, res)

    # time O(4^n / sqrt(n))  -> the n-th catalan number
    # space O(4^n / sqrt(n))


# -----------------------------------------------------------------------
"""
347. Top K Frequent Elements

Given a non-empty array of integers, return the k most frequent elements.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
Note:

You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.

"""

import heapq
import collections


class Solution:
    def topKFrequent(self, nums, k):
        H = []
        count = collections.Counter(nums)

        for num in count:
            if len(H) == k:
                curr = heapq.heappop(H)
                if curr[0] < count[num]:
                    curr = (count[num], num)
                heapq.heappush(H, curr)
            else:
                heapq.heappush(H, (count[num], num))

        return [val[1] for val in H]

        # time O(n log(k))
        # space O(n)


# -----------------------------------------------------------------------
"""
238. Product of Array Except Self

Given an array nums of n integers where n > 1,  return an array output such 
that output[i] is equal to the product of all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Constraint: It's guaranteed that the product of the elements of any prefix or 
suffix of the array (including the whole array) fits in a 32 bit integer.

"""


class Solution:
    def productExceptSelf(self, nums):
        allProd = 1
        zeros = 0
        for val in nums:
            if val != 0:
                allProd *= val
            else:
                zeros += 1

        if zeros > 1:
            return [0] * len(nums)

        for i in range(len(nums)):
            if nums[i] != 0:
                if zeros == 1:
                    currProd = 0
                else:
                    currProd = allProd // nums[i]
            else:
                currProd = allProd

            nums[i] = currProd

        return nums

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
78. Subsets

Given a set of distinct integers, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

Example:

Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

"""


class Solution:
    def subsets(self, nums):
        if len(nums) == 0:
            return [[]]

        res = [[]]

        for num in nums:
            curr = [val for val in res]
            temp = []
            for val in curr:
                temp.append([v for v in val])
                temp.append(val + [num])

            res = temp

        return res

        # time O(2^n)
        # space O(2^n)


# -----------------------------------------------------------------------
"""
230. Kth Smallest Element in a BST

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note:
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Example 1:

Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1


"""


class Solution:
    def kthSmallest(self, root, k):

        inOrder = self.inOrder(root)

        if not inOrder:
            return -1

        return inOrder[k - 1]

        # time O(n)
        # space O(n)

    def inOrder(self, node):

        if not node:
            return []

        left = self.inOrder(node.left)
        right = self.inOrder(node.right)

        return left + [node.val] + right


# -----------------------------------------------------------------------
"""
49. Group Anagrams

Given an array of strings, group anagrams together.

Example:

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
Note:

All inputs will be in lowercase.
The order of your output does not matter.
"""


class Solution:
    def groupAnagrams(self, strs):
        groups = {}

        for word in strs:
            curr = sorted(word)
            curr = ''.join(curr)
            if curr in groups:
                groups[curr].append(word)
            else:
                groups[curr] = [word]

        res = []
        for word in groups:
            res.append(groups[word])

        return res

        # time O(n * k * log(k)) -> k the longest word
        # space O(n * k)


# -----------------------------------------------------------------------
"""
48. Rotate Image

You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

Note:

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:

Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

"""


class Solution:
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """

        for row in range(0, len(matrix) // 2):
            for col in range(row, len(matrix) - row - 1):
                self.makeRotation(matrix, row, col)

        # time O(n^2)
        # space O(1)

    def makeRotation(self, arr, row, col):
        topleft = arr[row][col]
        arr[row][col] = arr[len(arr) - col - 1][row]  # bottom left to top right
        arr[len(arr) - col - 1][row] = arr[len(arr) - row - 1][len(arr) - col - 1]  # bottom right to bottom left
        arr[len(arr) - row - 1][len(arr) - col - 1] = arr[col][len(arr) - row - 1]  # top right to bottom left
        arr[col][len(arr) - row - 1] = topleft  # top left to top right


# -----------------------------------------------------------------------
"""
215. Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:

Input: [3,2,1,5,6,4] and k = 2
Output: 5

"""


class Solution:
    def findKthLargest(self, nums, k):
        H = []

        for num in nums:
            if len(H) == k:
                curr = heapq.heappop(H)
                if num > curr:
                    curr = num
                heapq.heappush(H, curr)
            else:
                heapq.heappush(H, num)

        return heapq.heappop(H)

    # time O(n * log(k))
    # space O(k)


# -----------------------------------------------------------------------
"""
102. Binary Tree Level Order Traversal

Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
"""


class Solution:
    def levelOrder(self, root):

        if not root:
            return []

        q = [root]
        res = []
        while len(q) > 0:

            length = len(q)
            currRes = []
            while length > 0:
                curr = q.pop(0)

                currRes.append(curr.val)

                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)

                length -= 1

            res.append(currRes)

        return res

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def addTwoNumbers(self, l1, l2):

        p1, p2 = l1, l2
        Sum = 0
        carry = 0

        DM = ListNode('DUMMY')
        dp = DM

        while p1 and p2:
            Sum = p1.val + p2.val + carry

            carry = Sum // 10
            Sum = Sum % 10

            node = ListNode(Sum)
            dp.next = node
            dp = dp.next
            p1 = p1.next
            p2 = p2.next

        while p1:
            Sum = p1.val + carry
            carry = Sum // 10
            Sum = Sum % 10

            node = ListNode(Sum)
            dp.next = node
            dp = dp.next
            p1 = p1.next

        while p2:
            Sum = p2.val + carry
            carry = Sum // 10
            Sum = Sum % 10

            node = ListNode(Sum)
            dp.next = node
            dp = dp.next
            p2 = p2.next

        if carry != 0:
            node = ListNode(carry)
            dp.next = node
            dp = dp.next

        return DM.next

        # time O(n + m)
        # space O(max(n,m))


# -----------------------------------------------------------------------
"""
378. Kth Smallest Element in a Sorted Matrix

Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:

matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
Note:
You may assume k is always valid, 1 ≤ k ≤ n2.
"""


class Solution:
    def kthSmallest(self, matrix, k):
        if len(matrix) == 0:
            return -1
        H = []

        for row in range(len(matrix)):
            for col in range(len(matrix)):
                if len(H) == k:
                    curr = heapq.heappop(H)
                    if curr < -1 * matrix[row][col]:
                        curr = -1 * matrix[row][col]
                    heapq.heappush(H, curr)
                else:
                    heapq.heappush(H, -1 * matrix[row][col])

        return heapq.heappop(H) * -1

        # time O(n^2 * log(k))
        # space O(k)


# -----------------------------------------------------------------------
"""
328. Odd Even Linked List

Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example 1:

Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL

Note:

The relative order inside both the even and odd groups should remain as it was in the input.
The first node is considered odd, the second node even and so on ...

"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def oddEvenList(self, head):

        EDN = ListNode('DUMMY')
        ODN = ListNode('DUMMY')

        pEven, pOdd, p = EDN, ODN, head

        switch = 0
        while p:
            next = p.next

            if switch == 0:
                pOdd.next = p
                pOdd = pOdd.next
                pOdd.next = None
                p = next
                switch = 1
            else:
                pEven.next = p
                pEven = pEven.next
                pEven.next = None
                p = next
                switch = 0
        pOdd.next = EDN.next

        return ODN.next

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
454. 4Sum II

Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.

Example:

Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2

Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0

"""


class Solution:
    def fourSumCount(self, A, B, C, D):

        counter = 0
        CDPaires = {}
        for c in C:
            for d in D:
                curr = (c + d) * -1
                if curr in CDPaires:
                    CDPaires[curr] += 1
                else:
                    CDPaires[curr] = 1

        for p1 in range(len(A)):
            for p2 in range(len(B)):
                curr = A[p1] + B[p2]
                if curr in CDPaires:
                    counter += CDPaires[curr]

        return counter

        # time O(max(a*b, c*d))
        # space O(c*d)


# -----------------------------------------------------------------------
"""
384. Shuffle an Array

Shuffle a set of numbers without duplicates.

Example:

// Init an array with set 1, 2, and 3.
int[] nums = {1,2,3};
Solution solution = new Solution(nums);

// Shuffle the array [1,2,3] and return its result. Any permutation of [1,2,3] must equally likely to be returned.
solution.shuffle();

// Resets the array back to its original configuration [1,2,3].
solution.reset();

// Returns the random shuffling of array [1,2,3].
solution.shuffle();

"""

import random


class Solution:

    def __init__(self, nums):
        self.original = list(nums)
        self.array = list(nums)

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        """
        self.array = list(self.original)
        return self.array

        # time O(n)
        # space O(1)

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        """

        for i in range(len(self.array)):
            newIdx = random.randrange(i, len(self.array))
            self.array[i], self.array[newIdx] = self.array[newIdx], self.array[i]

        return self.array

        # time O(n)
        # space O(1)


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()

# -----------------------------------------------------------------------
"""
289. Game of Life

According to the Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

Given a board with m by n cells, each cell has an initial state live (1) or dead (0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

Any live cell with fewer than two live neighbors dies, as if caused by under-population.
Any live cell with two or three live neighbors lives on to the next generation.
Any live cell with more than three live neighbors dies, as if by over-population..
Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
Write a function to compute the next state (after one update) of the board given its current state. The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously.

Example:

Input: 
[
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]
Output: 
[
  [0,0,0],
  [1,0,1],
  [0,1,1],
  [0,1,0]
]
Follow up:

Could you solve it in-place? Remember that the board needs to be updated at the 
same time: You cannot update some cells first and then use their updated 
values to update other cells.
In this question, we represent the board using a 2D array. In principle, 
the board is infinite, which would cause problems when the active area encroaches
the border of the array. How would you address these problems?
"""


class Solution:
    def gameOfLife(self, board):
        """
        Do not return anything, modify board in-place instead.
        """

        for row in range(len(board)):
            for col in range(len(board[0])):
                self.applyRules(board, row, col)

        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == 2:
                    board[row][col] = 0
                elif board[row][col] == -1:
                    board[row][col] = 1
        # time O(n * m)
        # space O(1)

    def applyRules(self, mat, row, col):
        numOfLives = self.numOfLiveAround(mat, row, col)

        if mat[row][col] == 1:
            if numOfLives < 2:
                mat[row][col] = 2
            elif numOfLives == 2 or numOfLives == 3:
                return
            else:
                mat[row][col] = 2
        else:
            if numOfLives == 3:
                mat[row][col] = -1

    def numOfLiveAround(self, mat, row, col):
        r_dx = [-1, 0, 1]
        c_dx = [-1, 0, 1]

        count = 0
        for r in r_dx:
            for c in c_dx:
                if r != 0 or c != 0:
                    if len(mat) > row + r >= 0 and \
                            len(mat[0]) > col + c >= 0:
                        if mat[row + r][col + c] >= 1:
                            count += 1

        return count


# -----------------------------------------------------------------------
"""
62. Unique Paths

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?


Above is a 7 x 3 grid. How many possible unique paths are there?

 

Example 1:

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
Example 2:

Input: m = 7, n = 3
Output: 28

"""


class Solution:
    def uniquePaths(self, m, n):

        memo = [[0 for j in range(n)] for i in range(m)]

        memo[0][0] = 1

        for row in range(m):
            memo[row][0] = 1

        for col in range(n):
            memo[0][col] = 1

        for row in range(1, m):
            for col in range(1, n):
                memo[row][col] = memo[row - 1][col] + memo[row][col - 1]

        return memo[m - 1][n - 1]

        # time O(m * n)
        # space O(m * n)

# -----------------------------------------------------------------------
"""

"""
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
