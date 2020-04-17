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
        arr[row][col] = arr[len(arr) - col - 1][row]  # bottom left to top left
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
11. Container With Most Water

Given n non-negative integers a1, a2, ..., an , where each represents a point at
coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). 
Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

Example:

Input: [1,8,6,2,5,4,8,3,7]
Output: 49

"""


class Solution:
    def maxArea(self, height):
        maxArea = 0
        left, right = 0, len(height) - 1

        while left < right:
            h = min(height[left], height[right])

            currArea = h * (right - left)

            if currArea > maxArea:
                maxArea = currArea

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return maxArea

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
36. Valid Sudoku

Determine if a 9x9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the 9 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition.

The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
Example 1:

Input:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: true

"""


class Solution:
    def isValidSudoku(self, board):
        rows = {}
        cols = {}

        for row in range(len(board)):
            for col in range(len(board[0])):
                currNum = board[row][col]

                if currNum == '.':
                    continue

                if currNum < '1' or currNum > '9':
                    return False

                if not self.checkIfExist(currNum, row, rows):
                    return False

                if not self.checkIfExist(currNum, col, cols):
                    return False

            if (row + 1) % 3 == 0:
                if not self.checkSquares(row - 2, row, board):
                    return False

        return True

        # time O(n^2)
        # space O(n^2)

    def checkSquares(self, f, t, board):

        for squareNum in range(3):
            currNums = []
            for row in range(f, t + 1):
                for col in range(squareNum * 3, (squareNum * 3) + 3):
                    if board[row][col] == '.':
                        continue
                    if board[row][col] in currNums:
                        return False
                    currNums.append(board[row][col])
        return True

    def checkIfExist(self, num, idx, hashMap):
        if not idx in hashMap:
            hashMap[idx] = [num]
        else:
            if num in hashMap[idx]:
                return False
            else:
                hashMap[idx].append(num)

        return True


# -----------------------------------------------------------------------
"""
105. Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7

"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder, inorder):

        if not preorder or not inorder:
            return None

        rootData = preorder[0]
        rootIdx = self.getRootIdx(inorder, rootData)
        node = TreeNode(rootData)

        node.left = self.buildTree(preorder[1: rootIdx + 1], inorder[: rootIdx])
        node.right = self.buildTree(preorder[rootIdx + 1:], inorder[rootIdx + 1:])

        return node

        # time O(n)
        # space O(n)

    def getRootIdx(self, inOrder, rootData):
        for i in range(len(inOrder)):
            if inOrder[i] == rootData:
                return i

        return -1


# -----------------------------------------------------------------------
"""
103. Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]

"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def zigzagLevelOrder(self, root):

        if not root:
            return

        q = [root]
        zigZag = False
        res = []
        while len(q) > 0:
            length = len(q)
            res.append(self.getLevel(q, zigZag))
            if zigZag:
                zigZag = False
            else:
                zigZag = True

            while length > 0:
                curr = q.pop(0)
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)

                length -= 1

        return res

        # time O(n)
        # space O(n)

    def getLevel(self, q, flag):
        currList = [q[i].val for i in range(len(q))]
        if flag:
            currList.reverse()
        return currList


# -----------------------------------------------------------------------
"""
380. Insert Delete GetRandom O(1)

Design a data structure that supports all following operations in average O(1) time.

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom(): Returns a random element from current set of elements. Each element must have the same probability of being returned.
Example:

// Init an empty set.
RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomSet.insert(1);

// Returns false as 2 does not exist in the set.
randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
randomSet.remove(1);

// 2 was already in the set, so return false.
randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
randomSet.getRandom();
"""


class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.map = {}
        self.array = []

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.map:
            return False

        self.array.append(val)
        self.map[val] = len(self.array) - 1

        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.map:
            return False

        if len(self.array) == 1:
            self.array = []
            self.map = {}
            return True

        currIdx = self.map[val]
        if currIdx == len(self.array) - 1:
            self.array.pop()
            del self.map[val]
            return True

        lastVal = self.array[len(self.array) - 1]
        self.array[currIdx] = lastVal
        self.array.pop()
        del self.map[val]
        self.map[lastVal] = currIdx
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """

        idx = random.randrange(0, len(self.array))

        return self.array[idx]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()


# -----------------------------------------------------------------------
"""
617. Merge Two Binary Trees

Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

Example 1:

Input: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7

"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mergeTrees(self, t1, t2):
        if not t1 and not t2:
            return None

        if not t1:
            return t2

        if not t2:
            return t1

        currNode = TreeNode(t1.val + t2.val)

        currNode.left = self.mergeTrees(t1.left, t2.left)
        currNode.right = self.mergeTrees(t1.right, t2.right)

        return currNode

        # time O(n + m)
        # space O(n + m)


# -----------------------------------------------------------------------
"""
942. DI String Match

Given a string S that only contains "I" (increase) or "D" (decrease), let N = S.length.

Return any permutation A of [0, 1, ..., N] such that for all i = 0, ..., N-1:

If S[i] == "I", then A[i] < A[i+1]
If S[i] == "D", then A[i] > A[i+1]
 

Example 1:

Input: "IDID"
Output: [0,4,1,3,2]
Example 2:

Input: "III"
Output: [0,1,2,3]
Example 3:

Input: "DDI"
Output: [3,2,0,1]

"""


class Solution:
    def diStringMatch(self, S):
        res = [0] * (len(S) + 1)
        left, right = 0, len(S)

        for i in range(len(S)):
            if S[i] == 'I':
                res[i] = left
                left += 1
            else:
                res[i] = right
                right -= 1

        res[len(res) - 1] = left

        return res

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
461. Hamming Distance

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers x and y, calculate the Hamming distance.

Note:
0 ≤ x, y < 231.

Example:

Input: x = 1, y = 4

Output: 2

Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

The above arrows point to positions where the corresponding bits are different.
"""


class Solution:
    def hammingDistance(self, x, y):

        res = x ^ y
        count = 0
        while res > 0:
            if res & 1 != 0:
                count += 1
            res >>= 1

        return count

        # time O(log(n) + log(m))
        # space O(log(n) + log(m))


# -----------------------------------------------------------------------
"""
1207. Unique Number of Occurrences

Given an array of integers arr, write a function that returns true if and only if the number of occurrences of each value in the array is unique.

 

Example 1:

Input: arr = [1,2,2,1,1,3]
Output: true
Explanation: The value 1 has 3 occurrences, 2 has 2 and 3 has 1. No two values have the same number of occurrences.
Example 2:

Input: arr = [1,2]
Output: false
Example 3:

Input: arr = [-3,0,1,-3,1,1,1,-3,10,0]
Output: true

"""


class Solution:
    def uniqueOccurrences(self, arr):
        counter = collections.Counter(arr)

        seen = set()

        for num in counter:
            if counter[num] in seen:
                return False
            seen.add(counter[num])

        return True

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
1385. Find the Distance Value Between Two Arrays

Given two integer arrays arr1 and arr2, and the integer d, return the distance value between the two arrays.

The distance value is defined as the number of elements arr1[i] such that there is not any element arr2[j] where |arr1[i]-arr2[j]| <= d.

 

Example 1:

Input: arr1 = [4,5,8], arr2 = [10,9,1,8], d = 2
Output: 2
Explanation: 
For arr1[0]=4 we have: 
|4-10|=6 > d=2 
|4-9|=5 > d=2 
|4-1|=3 > d=2 
|4-8|=4 > d=2 
For arr1[1]=5 we have: 
|5-10|=5 > d=2 
|5-9|=4 > d=2 
|5-1|=4 > d=2 
|5-8|=3 > d=2
For arr1[2]=8 we have:
|8-10|=2 <= d=2
|8-9|=1 <= d=2
|8-1|=7 > d=2
|8-8|=0 <= d=2

"""


class Solution:
    def findTheDistanceValue(self, arr1, arr2, d):
        counter = 0

        for a in arr1:
            flag = True
            for b in arr2:
                if abs(a - b) <= d:
                    flag = False
                    break

            if flag:
                counter += 1

        return counter

        # time O(n*m)
        # space O(1)


# -----------------------------------------------------------------------
"""
852. Peak Index in a Mountain Array

Let's call an array A a mountain if the following properties hold:

A.length >= 3
There exists some 0 < i < A.length - 1 such that A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1]
Given an array that is definitely a mountain, return any i such that A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1].

Example 1:

Input: [0,1,0]
Output: 1

"""


class Solution:
    def peakIndexInMountainArray(self, A):
        if len(A) < 3:
            return -1

        maxNum = max(A)

        for i in range(1, len(A) - 1):
            if A[i] == maxNum:
                return i

        return -1

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
590. N-ary Tree Postorder Traversal

Given an n-ary tree, return the postorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).

 

Follow up:

Recursive solution is trivial, could you do it iteratively?

 

Example 1:

Input: root = [1,null,3,2,4,null,5,6]
Output: [5,6,3,2,4,1]
"""

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


class Solution:
    def postorder(self, root):

        if not root:
            return []
        res = []
        for c in root.children:
            res += self.postorder(c)

        res += [root.val]

        return res

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
589. N-ary Tree Preorder Traversal

Given an n-ary tree, return the preorder traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).

 

Follow up:

Recursive solution is trivial, could you do it iteratively?

 

Example 1:

Input: root = [1,null,3,2,4,null,5,6]
Output: [1,3,5,6,2,4]

"""

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


class Solution:
    def preorder(self, root):
        if not root:
            return []

        res = [root.val]

        for c in root.children:
            res += self.preorder(c)

        return res

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
1342. Number of Steps to Reduce a Number to Zero

Given a non-negative integer num, return the number of steps to reduce it to zero. If the current number is even, you have to divide it by 2, otherwise, you have to subtract 1 from it.

 

Example 1:

Input: num = 14
Output: 6
Explanation: 
Step 1) 14 is even; divide by 2 and obtain 7. 
Step 2) 7 is odd; subtract 1 and obtain 6.
Step 3) 6 is even; divide by 2 and obtain 3. 
Step 4) 3 is odd; subtract 1 and obtain 2. 
Step 5) 2 is even; divide by 2 and obtain 1. 
Step 6) 1 is odd; subtract 1 and obtain 0.

"""


class Solution:
    def numberOfSteps(self, num):

        counter = 0
        while num > 0:
            if num % 2 == 0:
                num //= 2
            else:
                num -= 1

            counter += 1

        return counter

        # time O(log(n))
        # space O(1)


# -----------------------------------------------------------------------
"""
1365. How Many Numbers Are Smaller Than the Current Number

Given the array nums, for each nums[i] find out how many numbers in the array 
are smaller than it. That is, for each nums[i] you have to count the number of valid j's such that 
j != i and nums[j] < nums[i].

Return the answer in an array.
"""


class Solution:
    def smallerNumbersThanCurrent(self, nums):
        res = []
        for i in range(len(nums)):
            counter = 0
            for j in range(len(nums)):
                if i != j:
                    if nums[j] < nums[i]:
                        counter += 1
            res.append(counter)

        return res


# -----------------------------------------------------------------------
"""
1313. Decompress Run-Length Encoded List

We are given a list nums of integers representing a list compressed with run-length encoding.

Consider each adjacent pair of elements [freq, val] = [nums[2*i], nums[2*i+1]] (with i >= 0).  
For each such pair, there are freq elements with value val concatenated in a sublist. Concatenate all the 
sublists from left to right to generate the decompressed list.

Return the decompressed list.
"""


class Solution:
    def decompressRLElist(self, nums):
        res = []

        for i in range(0, len(nums) - 1, 2):
            freq = nums[i]
            val = nums[i + 1]
            res.extend([val] * freq)

        return res


# -----------------------------------------------------------------------
"""
1295. Find Numbers with Even Number of Digits

Given an array nums of integers, return how many of them contain an even number of digits.
 

Example 1:

Input: nums = [12,345,2,6,7896]
Output: 2
Explanation: 
12 contains 2 digits (even number of digits). 
345 contains 3 digits (odd number of digits). 
2 contains 1 digit (odd number of digits). 
6 contains 1 digit (odd number of digits). 
7896 contains 4 digits (even number of digits). 
Therefore only 12 and 7896 contain an even number of digits.

"""


class Solution:
    def findNumbers(self, nums):
        counter = 0

        for num in nums:
            if self.countDigits(num) % 2 == 0:
                counter += 1

        return counter

        # time O(n * log(m)) -> m num of digits of the largest number
        # space O(1)

    def countDigits(self, num):
        counter = 0
        while num > 0:
            counter += 1
            num //= 10

        return counter


# -----------------------------------------------------------------------
"""
1281. Subtract the Product and Sum of Digits of an Integer

Given an integer number n, return the difference between the product of its digits and the sum of its digits.
 

Example 1:

Input: n = 234
Output: 15 
Explanation: 
Product of digits = 2 * 3 * 4 = 24 
Sum of digits = 2 + 3 + 4 = 9 
Result = 24 - 9 = 15
"""


class Solution:
    def subtractProductAndSum(self, n):

        digits = self.getDigits(n)
        Sum = sum(digits)
        prod = 1
        for d in digits:
            prod *= d

        return prod - Sum

        # time O(log(n))
        # space O(log(n))

    def getDigits(self, n):
        res = []

        while n > 0:
            res.append(n % 10)
            n //= 10

        return res


# -----------------------------------------------------------------------
"""
1002. Find Common Characters

Given an array A of strings made only from lowercase letters, return a list of all characters that show up in all strings within the list (including duplicates).  For example, if a character occurs 3 times in all strings but not 4 times, you need to include that character three times in the final answer.

You may return the answer in any order.

 

Example 1:

Input: ["bella","label","roller"]
Output: ["e","l","l"]

"""


class Solution:
    def commonChars(self, A):
        counter = []

        for s in A:
            counter.append(collections.Counter(s))

        res = counter[0]

        for currCount in counter:
            deleted = []
            for char in res:
                if char not in currCount:
                    deleted.append(char)
                else:
                    res[char] = min(res[char], currCount[char])

            for char in deleted:
                del res[char]

        ans = []

        for char in res:
            ans.extend([char] * res[char])

        return ans

        # time O(n * l) -> l the length of largest string
        # space O(n * l)


# -----------------------------------------------------------------------
"""
1021. Remove Outermost Parentheses

A valid parentheses string is either empty (""), "(" + A + ")", or A + B, where A and B are valid parentheses strings, and + represents string concatenation.  For example, "", "()", "(())()", and "(()(()))" are all valid parentheses strings.

A valid parentheses string S is primitive if it is nonempty, and there does not exist a way to split it into S = A+B, with A and B nonempty valid parentheses strings.

Given a valid parentheses string S, consider its primitive decomposition: S = P_1 + P_2 + ... + P_k, where P_i are primitive valid parentheses strings.

Return S after removing the outermost parentheses of every primitive string in the primitive decomposition of S.

 

Example 1:

Input: "(()())(())"
Output: "()()()"
Explanation: 
The input string is "(()())(())", with primitive decomposition "(()())" + "(())".
After removing outer parentheses of each part, this is "()()" + "()" = "()()()".
"""


class Solution:
    def removeOuterParentheses(self, S: str) -> str:
        premitive = []

        currCounter = 0
        curr = ''
        for c in S:
            if c == '(':
                currCounter += 1
                curr += c
            else:
                currCounter -= 1
                curr += c

            if currCounter == 0:
                premitive.append(curr)
                curr = ''

        res = ''
        for val in premitive:
            for i in range(1, len(val) - 1):
                res += val[i]

        return res

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
1108. Defanging an IP Address

Given a valid (IPv4) IP address, return a defanged version of that IP address.

A defanged IP address replaces every period "." with "[.]".

Example 1:

Input: address = "1.1.1.1"
Output: "1[.]1[.]1[.]1"
"""


class Solution:
    def defangIPaddr(self, address):
        res = []
        for i in range(len(address)):
            if address[i] == '.':
                res.append('[.]')
            else:
                res.append(address[i])
        return ''.join(res)

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
1389. Create Target Array in the Given Order

Given two arrays of integers nums and index. Your task is to create target array under the following rules:

Initially target array is empty.
From left to right read nums[i] and index[i], insert at index index[i] the value nums[i] in target array.
Repeat the previous step until there are no elements to read in nums and index.
Return the target array.

It is guaranteed that the insertion operations will be valid.

 

Example 1:

Input: nums = [0,1,2,3,4], index = [0,1,2,2,1]
Output: [0,4,1,3,2]
Explanation:
nums       index     target
0            0        [0]
1            1        [0,1]
2            2        [0,1,2]
3            2        [0,1,3,2]
4            1        [0,4,1,3,2]

"""


class Solution:
    def createTargetArray(self, nums, index):
        target = []

        i = 0

        while i < len(nums) and i < len(index):
            idx = index[i]
            val = nums[i]
            target.insert(idx, val)
            i += 1

        return target

        # time O(n^2)
        # space O(n)


# -----------------------------------------------------------------------
"""
1221. Split a String in Balanced Strings

Balanced strings are those who have equal quantity of 'L' and 'R' characters.

Given a balanced string s split it in the maximum amount of balanced strings.

Return the maximum amount of splitted balanced strings.

 

Example 1:

Input: s = "RLRRLLRLRL"
Output: 4
Explanation: s can be split into "RL", "RRLL", "RL", "RL", each substring contains same number of 'L' and 'R'.

"""


class Solution:
    def balancedStringSplit(self, s):

        res, counter, curr = 0, 0, 0

        while curr < len(s):
            if s[curr] == 'L':
                counter += 1
            else:
                counter -= 1

            if counter == 0:
                res += 1

            curr += 1

        return res

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1290. Convert Binary Number in a Linked List to Integer

Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1.
The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.

 

Example 1:


Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10
"""


# -----------------------------------------------------------------------
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getDecimalValue(self, head):

        power, res, curr = 0, 0, head

        while curr:
            curr = curr.next
            power += 1

        power -= 1
        curr = head

        while curr:
            res += curr.val * pow(2, power)
            power -= 1
            curr = curr.next

        return res

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1266. Minimum Time Visiting All Points

On a plane there are n points with integer coordinates points[i] = [xi, yi]. Your task is to find the minimum time
in seconds to visit all points.

You can move according to the next rules:

In one second always you can either move vertically, horizontally by one unit or diagonally 
(it means to move one unit vertically and one unit horizontally in one second).
You have to visit the points in the same order as they appear in the array.
 

Example 1:


Input: points = [[1,1],[3,4],[-1,0]]
Output: 7
Explanation: One optimal path is [1,1] -> [2,2] -> [3,3] -> [3,4] -> [2,3] -> [1,2] -> [0,1] -> [-1,0]   
Time from [1,1] to [3,4] = 3 seconds 
Time from [3,4] to [-1,0] = 4 seconds
Total time = 7 seconds
"""


class Solution:
    def minTimeToVisitAllPoints(self, points):
        minTime = 0

        for i in range(1, len(points)):
            dx = abs(points[i][0] - points[i - 1][0])
            dy = abs(points[i][1] - points[i - 1][1])
            minTime += max(dx, dy)

        return minTime

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1299. Replace Elements with Greatest Element on Right Side

Given an array arr, replace every element in that array with the greatest element among the 
elements to its right, and replace the last element with -1.

After doing so, return the array.

 

Example 1:

Input: arr = [17,18,5,4,6,1]
Output: [18,6,6,6,1,-1]
"""


class Solution:
    def replaceElements(self, arr):

        maxNum = arr[len(arr) - 1]
        arr[len(arr) - 1] = -1
        for i in range(len(arr) - 2, -1, -1):
            currNum = arr[i]
            arr[i] = maxNum
            if currNum > maxNum:
                maxNum = currNum

        return arr

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1323. Maximum 69 Number

Given a positive integer num consisting only of digits 6 and 9.

Return the maximum number you can get by changing at most one digit (6 becomes 9, and 9 becomes 6).

 

Example 1:

Input: num = 9669
Output: 9969
Explanation: 
Changing the first digit results in 6669.
Changing the second digit results in 9969.
Changing the third digit results in 9699.
Changing the fourth digit results in 9666. 
The maximum number is 9969.
Example 2:

Input: num = 9996
Output: 9999
Explanation: Changing the last digit 6 to 9 results in the maximum number.

"""


class Solution:
    def maximum69Number(self, num):

        digits = self.getDigits(num)

        flag = False
        res = 0

        for i in range(len(digits) - 1, -1, -1):
            if digits[i] == 6 and not flag:
                digits[i] = 9
                flag = True

            res = res * 10 + digits[i]

        return res

        # time O(log(n))
        # space O(log(n))

    def getDigits(self, num):

        digits = []

        while num > 0:
            digits.append(num % 10)
            num //= 10

        return digits


# -----------------------------------------------------------------------
"""
1351. Count Negative Numbers in a Sorted Matrix

Given a m * n matrix grid which is sorted in non-increasing order both row-wise and column-wise. 

Return the number of negative numbers in grid.

 

Example 1:

Input: grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]
Output: 8
Explanation: There are 8 negatives number in the matrix.

Example 2:

Input: grid = [[3,2],[1,0]]
Output: 0
"""


class Solution:
    def countNegatives(self, grid):

        res = 0
        for row in grid:
            res += self.countNegs(row)

        return res

        # time O(m * log(n))
        # space O(1)

    def countNegs(self, arr):
        start, end = 0, len(arr) - 1

        if arr[end] >= 0:
            return 0

        if arr[start] < 0:
            return len(arr)

        while start <= end:
            mid = (start + end) // 2
            if arr[mid] < 0:
                if arr[mid - 1] >= 0:
                    return len(arr) - mid
                else:
                    end = mid - 1
            else:
                start = mid + 1

        return -1


# -----------------------------------------------------------------------
"""
1252. Cells with Odd Values in a Matrix

Given n and m which are the dimensions of a matrix initialized by zeros and given an array indices where
indices[i] = [ri, ci]. For each pair of [ri, ci] you have to increment all cells in row ri and column ci by 1.

Return the number of cells with odd values in the matrix after applying the increment to all indices.

 

Example 1:


Input: n = 2, m = 3, indices = [[0,1],[1,1]]
Output: 6
Explanation: Initial matrix = [[0,0,0],[0,0,0]].
After applying first increment it becomes [[1,2,1],[0,1,0]].
The final matrix will be [[1,3,1],[1,3,1]] which contains 6 odd numbers.

"""


class Solution:
    def oddCells(self, n, m, indices):

        mat = [[0 for j in range(m)] for i in range(n)]
        res = 0
        for pair in indices:
            row = pair[0]
            col = pair[1]

            res = self.increment(mat, row, 'row', res)
            res = self.increment(mat, col, 'col', res)

        return res

        # time O(k * (m+n))
        # space O(m+n)

    def increment(self, mat, idx, flag, counter):
        if flag == 'row':
            for col in range(len(mat[0])):
                if mat[idx][col] % 2 == 0:
                    counter += 1
                else:
                    counter -= 1
                mat[idx][col] += 1
        else:
            for row in range(len(mat)):
                if mat[row][idx] % 2 == 0:
                    counter += 1
                else:
                    counter -= 1
                mat[row][idx] += 1

        return counter


# -----------------------------------------------------------------------
"""
1370. Increasing Decreasing String

Given a string s. You should re-order the string using the following algorithm:

Pick the smallest character from s and append it to the result.
Pick the smallest character from s which is greater than the last appended character to the result and append it.
Repeat step 2 until you cannot pick more characters.
Pick the largest character from s and append it to the result.
Pick the largest character from s which is smaller than the last appended character to the result and append it.
Repeat step 5 until you cannot pick more characters.
Repeat the steps from 1 to 6 until you pick all characters from s.
In each step, If the smallest or the largest character appears more than once you can choose any occurrence 
and append it to the result.

Return the result string after sorting s with this algorithm.

 

Example 1:

Input: s = "aaaabbbbcccc"
Output: "abccbaabccba"
Explanation: After steps 1, 2 and 3 of the first iteration, result = "abc"
After steps 4, 5 and 6 of the first iteration, result = "abccba"
First iteration is done. Now s = "aabbcc" and we go back to step 1
After steps 1, 2 and 3 of the second iteration, result = "abccbaabc"
After steps 4, 5 and 6 of the second iteration, result = "abccbaabccba"

"""


class Solution:
    def sortString(self, s):

        count = collections.Counter(s)
        res = ''
        while len(count) > 0:

            for key in sorted(count):
                res += key
                count[key] -= 1
                if count[key] == 0:
                    del count[key]

            for key in sorted(count, reverse=True):
                res += key
                count[key] -= 1
                if count[key] == 0:
                    del count[key]

        return res

        # time O(c * k * log(k)) -> k : the number of distinct chars, c the counter of the maximum char
        # space O(n)


# -----------------------------------------------------------------------
"""
1309. Decrypt String from Alphabet to Integer Mapping

Given a string s formed by digits ('0' - '9') and '#' . We want to map s to English lowercase characters as follows:

Characters ('a' to 'i') are represented by ('1' to '9') respectively.
Characters ('j' to 'z') are represented by ('10#' to '26#') respectively. 
Return the string formed after mapping.

It's guaranteed that a unique mapping will always exist.

 

Example 1:

Input: s = "10#11#12"
Output: "jkab"
Explanation: "j" -> "10#" , "k" -> "11#" , "a" -> "1" , "b" -> "2".
"""


class Solution:
    def freqAlphabets(self, s):
        res = []

        for i in range(len(s)):
            if '0' <= s[i] <= '9':
                res.append(chr(int(s[i]) + 96))
            else:
                res.pop()
                res.pop()
                currNum = int(s[i - 2]) * 10 + int(s[i - 1])
                res.append(chr(currNum + 96))

        return ''.join(res)

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
1304. Find N Unique Integers Sum up to Zero

Given an integer n, return any array containing n unique integers such that they add up to 0.

Example 1:

Input: n = 5
Output: [-7,-1,1,3,4]
Explanation: These arrays also are accepted [-5,-1,1,2,3] , [-3,-1,2,-2,4].

Example 2:

Input: n = 3
Output: [-1,0,1]

"""


class Solution:
    def sumZero(self, n):
        if n == 1:
            return [0]

        start, end = 0, n - 1
        res = [0] * n
        currNum = n
        while start <= end:
            res[start] = currNum
            res[end] = -currNum
            currNum -= 1
            start += 1
            end -= 1

        if n % 2 != 0:
            res[n // 2] = 0

        return res

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
153. Find Minimum in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.

Example 1:

Input: [3,4,5,1,2] 
Output: 1

"""


class Solution:
    def findMin(self, nums):
        if len(nums) == 1:
            return nums[0]
        idx = self.getPivotIdx(nums)

        if idx == -1:
            return nums[0]

        return nums[idx + 1]

        # time O(log(n))
        # space O(1)

    def getPivotIdx(self, nums):
        start, end = 0, len(nums) - 1

        while start < end:
            mid = (start + end) // 2

            if nums[mid] > nums[mid + 1]:
                return mid
            elif nums[mid] >= nums[start]:
                start = mid + 1
            else:
                end = mid

        return -1


# -----------------------------------------------------------------------
"""
41. First Missing Positive

Given an unsorted integer array, find the smallest missing positive integer.

Example 1:

Input: [1,2,0]
Output: 3
Example 2:

Input: [3,4,-1,1]
Output: 2
Example 3:

Input: [7,8,9,11,12]
Output: 1
Note:

Your algorithm should run in O(n) time and uses constant extra space.
"""


class Solution:
    def firstMissingPositive(self, nums):
        n, inf = len(nums), len(nums) + 1

        if n == 0:
            return 1

        for i in range(n):
            if nums[i] <= 0:  # remove all elements less than or equal to zero
                nums[i] = inf

        for i in range(n):
            val = abs(nums[i])
            if val <= n and nums[val - 1] > 0:  # mark all elements between 1 to n as visited
                nums[val - 1] = -1 * nums[val - 1]

        for i in range(n):
            if nums[i] > 0:  # the first index not visited return its element
                return i + 1

        return n + 1  # if all elements between 1 to n are exist return  n + 1

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1380. Lucky Numbers in a Matrix

Given a m * n matrix of distinct numbers, return all lucky numbers in the matrix in any order.

A lucky number is an element of the matrix such that it is the minimum element in its row and maximum in its column.

 

Example 1:

Input: matrix = [[3,7,8],[9,11,13],[15,16,17]]
Output: [15]
Explanation: 15 is the only lucky number since it is the minimum in its row and the maximum in its column

"""


class Solution:
    def luckyNumbers(self, matrix):

        res = []

        minRows, maxCols = self.getMinMax(matrix)
        minRows = set(minRows)

        res = [val for val in maxCols if val in minRows]

        return res

        # time O(m * n)
        # space O(m + n)

    def getMinMax(self, mat):
        rows = [10 ** 15 for row in range(len(mat))]
        cols = [0 for col in range(len(mat[0]))]

        for row in range(len(mat)):
            for col in range(len(mat[0])):
                currVal = mat[row][col]
                rows[row] = min(rows[row], currVal)
                cols[col] = max(cols[col], currVal)

        return rows, cols


# -----------------------------------------------------------------------
"""
1394. Find Lucky Integer in an Array

Given an array of integers arr, a lucky integer is an integer which has a frequency in the array equal to its value.

Return a lucky integer in the array. If there are multiple lucky integers return the largest of them. If there is no lucky integer return -1.

 

Example 1:

Input: arr = [2,2,3,4]
Output: 2
Explanation: The only lucky number in the array is 2 because frequency[2] == 2.
"""


class Solution:
    def findLucky(self, arr):
        counter = collections.Counter(arr)

        lucky = -1

        for val in counter:
            if val == counter[val]:
                if val > lucky:
                    lucky = val

        return lucky

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
944. Delete Columns to Make Sorted

We are given an array A of N lowercase letter strings, all of the same length.

Now, we may choose any set of deletion indices, and for each string, we delete all the characters in those indices.

For example, if we have an array A = ["abcdef","uvwxyz"] and deletion indices {0, 2, 3}, then the final array after 
deletions is ["bef", "vyz"], and the remaining columns of A are ["b","v"], ["e","y"], and ["f","z"].  
(Formally, the c-th column is [A[0][c], A[1][c], ..., A[A.length-1][c]].)

Suppose we chose a set of deletion indices D such that after deletions, each remaining column in A is in 
non-decreasing sorted order.

Return the minimum possible value of D.length.

 

Example 1:

Input: ["cba","daf","ghi"]
Output: 1
Explanation: 
After choosing D = {1}, each column ["c","d","g"] and ["a","f","i"] are in non-decreasing sorted order.
If we chose D = {}, then a column ["b","a","h"] would not be in non-decreasing sorted order.
"""


class Solution:
    def minDeletionSize(self, A):

        res = 0

        for col in range(len(A[0])):
            for row in range(1, len(A)):
                if A[row - 1][col] > A[row][col]:
                    res += 1
                    break

        return res

        # time O(n * m)
        # space O(1)


# -----------------------------------------------------------------------
"""
1051. Height Checker

Students are asked to stand in non-decreasing order of heights for an annual photo.

Return the minimum number of students that must move in order for all students to be standing 
in non-decreasing order of height.

Notice that when a group of students is selected they can reorder in any possible way between 
themselves and the non selected students remain on their seats.

 

Example 1:

Input: heights = [1,1,4,2,1,3]
Output: 3
Explanation: 
Current array : [1,1,4,2,1,3]
Target array  : [1,1,1,2,3,4]
On index 2 (0-based) we have 4 vs 1 so we have to move this student.
On index 4 (0-based) we have 1 vs 3 so we have to move this student.
On index 5 (0-based) we have 3 vs 4 so we have to move this student.
"""


class Solution:
    def heightChecker(self, heights):
        target = sorted(heights)

        counter = 0
        for i in range(len(target)):
            if target[i] != heights[i]:
                counter += 1

        return counter

        # time O(n * log(n))
        # space O(n)


# -----------------------------------------------------------------------
"""
1356. Sort Integers by The Number of 1 Bits

Given an integer array arr. You have to sort the integers in the array in ascending order by the number of 1's 
in their binary representation and in case of two or more integers have the same number of 1's you have 
to sort them in ascending order.

Return the sorted array.

 

Example 1:

Input: arr = [0,1,2,3,4,5,6,7,8]
Output: [0,1,2,4,8,3,5,6,7]
Explantion: [0] is the only integer with 0 bits.
[1,2,4,8] all have 1 bit.
[3,5,6] have 2 bits.
[7] has 3 bits.
The sorted array by bits is [0,1,2,4,8,3,5,6,7]
"""


class Solution:
    def sortByBits(self, arr):
        arr.sort()
        for i in range(len(arr)):
            arr[i] = (arr[i], self.getNumOfOneBits(arr[i]))

        arr = sorted(arr, key=lambda pair: pair[1])
        arr = [pair[0] for pair in arr]

        return arr

        # time O(n * log(n))
        # space O(n)

    def getNumOfOneBits(self, num):
        counter = 0

        while num > 0:
            if num & 1 != 0:
                counter += 1
            num >>= 1

        return counter


# -----------------------------------------------------------------------
"""
268. Missing Number

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

Example 1:

Input: [3,0,1]
Output: 2

"""


class Solution:
    def missingNumber(self, nums):
        missing = 0

        for i in range(len(nums) + 1):
            missing ^= i

        for val in nums:
            missing ^= val

        return missing

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
131. Palindrome Partitioning

Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

Example:

Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
"""


class Solution:
    def partition(self, s):

        if len(s) == 0:
            return [[]]

        res = []
        currRes = []
        self.recurPartition(s, 0, res, currRes)

        return res

        # time O(2^n)
        # space O(n * 2^n)

    def recurPartition(self, s, idx, res, currRes):

        if idx >= len(s):
            res.append([val for val in currRes])
            return

        for i in range(idx + 1, len(s) + 1):
            currSub = s[idx:i]
            if self.isPal(currSub):
                currRes.append(currSub)
                self.recurPartition(s, i, res, currRes)
                currRes.pop()

    def isPal(self, s):
        rev = s[::-1]
        return rev == s


# -----------------------------------------------------------------------
"""
17. Letter Combinations of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

{2:['a','b','c'], 3:['d','e','f'], 4:['g','h','i'], 5:['j','k','l'], 6:['m','n','o'], 7:['p','q','r','s'], 8:['t','u','v'], 9:['w','x','y','z']}

Example:

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
"""
digitsMap = {2: ['a', 'b', 'c'], 3: ['d', 'e', 'f'], 4: ['g', 'h', 'i'], 5: ['j', 'k', 'l'], 6: ['m', 'n', 'o'],
             7: ['p', 'q', 'r', 's'], 8: ['t', 'u', 'v'], 9: ['w', 'x', 'y', 'z']}


class Solution:
    def letterCombinations(self, digits):
        if len(digits) == 0:
            return []
        res = []
        self.recurComb(digits, 0, [], res)
        return res

        # time O(3^n * 4^m) -> n num of digits has 3 chars, m num of digits has 4 chars
        # space O(3^n * 4^m)

    def recurComb(self, digits, idx, currRes, res):
        if idx >= len(digits):
            res.append(''.join(currRes))
            return

        currDig = digits[idx]
        for i in range(len(digitsMap[int(currDig)])):
            currChar = digitsMap[int(currDig)][i]
            currRes.append(currChar)
            self.recurComb(digits, idx + 1, currRes, res)
            currRes.pop()


# -----------------------------------------------------------------------
"""
42. Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it is able to trap after raining.

Example:

Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
"""


class Solution:

    #     def trap(self, height):
    #         if len(height) == 0:
    #             return 0
    #         n = len(height)
    #         leftMax, rightMax = [0] * n , [0] * n
    #         i, j = 0, len(height) - 1
    #         currLeftMax = height[0]
    #         currRightMax = height[len(height) - 1]
    #         while i < len(height):
    #             if height[i] > currLeftMax:
    #                 currLeftMax = height[i]

    #             if height[j] > currRightMax:
    #                 currRightMax = height[j]

    #             leftMax[i] = currLeftMax
    #             rightMax[j] = currRightMax
    #             i += 1
    #             j -= 1
    #         ans = 0
    #         for i in range(n):
    #             ans += (min(leftMax[i], rightMax[i]) - height[i])

    #         return ans

    #         # time O(n)
    #         # space O(n)

    def trap(self, height):

        left, right, lMax, rMax = 0, len(height) - 1, 0, 0
        ans = 0
        while left < right:
            if height[left] > lMax:
                lMax = height[left]
            if height[right] > rMax:
                rMax = height[right]

            if lMax < rMax:
                ans += lMax - height[left]
                left += 1
            else:
                ans += rMax - height[right]
                right -= 1

        return ans

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1337. The K Weakest Rows in a Matrix
Given a m * n matrix mat of ones (representing soldiers) and zeros (representing civilians), 
return the indexes of the k weakest rows in the matrix ordered from the weakest to the strongest.

A row i is weaker than row j, if the number of soldiers in row i is less than the number of soldiers in row j, 
or they have the same number of soldiers but i is less than j. Soldiers are always stand in the frontier of a row, 
that is, always ones may appear first and then zeros.

 

Example 1:

Input: mat = 
[[1,1,0,0,0],
 [1,1,1,1,0],
 [1,0,0,0,0],
 [1,1,0,0,0],
 [1,1,1,1,1]], 
k = 3
Output: [2,0,3]
"""


class Solution:
    def kWeakestRows(self, mat, k):

        maxH = []

        for row in range(len(mat)):
            soldsCount = self.numOfSold(mat, row) * -1
            if len(maxH) < k:
                heapq.heappush(maxH, (soldsCount, row * -1))
            else:
                curr = heapq.heappop(maxH)
                if soldsCount > curr[0]:
                    curr = (soldsCount, row * - 1)
                heapq.heappush(maxH, curr)
        maxH = [(val[0] * -1, val[1] * -1) for val in maxH]
        maxH = sorted(maxH, key=lambda x: (x[0], x[1]))
        return [val[1] for val in maxH]

        # time O(n)
        # space O(k)

    def numOfSold(self, mat, row):
        col = 0
        while col < len(mat[0]) and mat[row][col] == 1:
            col += 1

        return col


# -----------------------------------------------------------------------
"""
811. Subdomain Visit Count

A website domain like "discuss.leetcode.com" consists of various subdomains. At the top level, we have "com", 
at the next level, we have "leetcode.com", and at the lowest level, "discuss.leetcode.com". 
When we visit a domain like "discuss.leetcode.com", we will also visit the parent domains "leetcode.com" and "com" implicitly.

Now, call a "count-paired domain" to be a count (representing the number of visits this domain received), 
followed by a space, followed by the address. An example of a count-paired domain might be "9001 discuss.leetcode.com".

We are given a list cpdomains of count-paired domains. We would like a list of count-paired domains, 
(in the same format as the input, and in any order), that explicitly counts the number of visits to each subdomain.

Example 1:
Input: 
["9001 discuss.leetcode.com"]
Output: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
Explanation: 
We only have one website domain: "discuss.leetcode.com". As discussed above, the subdomain "leetcode.com" and 
"com" will also be visited. So they will all be visited 9001 times.
"""


class Solution:
    def subdomainVisits(self, cpdomains):
        count = {}

        for pair in cpdomains:
            visitCount, domain = pair.split()
            visitCount = int(visitCount)
            while domain != '':
                if domain in count:
                    count[domain] += visitCount
                else:
                    count[domain] = visitCount

                i = 0
                while i < len(domain) and domain[i] != '.':
                    i += 1
                domain = domain[i + 1:]

        res = []
        for domain in count:
            res.append('{} {}'.format(count[domain], domain))

        return res

        # time O(n * m)
        # space O(d) -> d num of domains


# -----------------------------------------------------------------------
"""
922. Sort Array By Parity II

Given an array A of non-negative integers, half of the integers in A are odd, and half of the integers are even.

Sort the array so that whenever A[i] is odd, i is odd; and whenever A[i] is even, i is even.

You may return any answer array that satisfies this condition.

 

Example 1:

Input: [4,2,5,7]
Output: [4,5,2,7]
Explanation: [4,7,2,5], [2,5,4,7], [2,7,4,5] would also have been accepted.

"""


class Solution:
    def sortArrayByParityII(self, A):

        firstOdd, firstEven = 1, 0

        while firstOdd < len(A) and firstEven < len(A):

            while firstOdd < len(A) and A[firstOdd] % 2 == 1:
                firstOdd += 2

            while firstEven < len(A) and A[firstEven] % 2 == 0:
                firstEven += 2

            if firstEven < len(A) and firstOdd < len(A):
                A[firstEven], A[firstOdd] = A[firstOdd], A[firstEven]
                firstOdd += 2
                firstEven += 2

        return A

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1237. Find Positive Integer Solution for a Given Equation

Given a function  f(x, y) and a value z, return all positive integer pairs x and y where f(x,y) == z.

The function is constantly increasing, i.e.:

f(x, y) < f(x + 1, y)
f(x, y) < f(x, y + 1)
The function interface is defined like this: 

interface CustomFunction {
public:
  // Returns positive integer f(x, y) for any given positive integer x and y.
  int f(int x, int y);
};
For custom testing purposes you're given an integer function_id and a target z as input, where function_id 
represent one function from an secret internal list, on the examples you'll know only two functions from the list.  

You may return the solutions in any order.


Example 1:

Input: function_id = 1, z = 5
Output: [[1,4],[2,3],[3,2],[4,1]]
Explanation: function_id = 1 means that f(x, y) = x + y

"""

"""
   This is the custom function interface.
   You should not implement it, or speculate about its implementation
   class CustomFunction:
       # Returns f(x, y) for any given positive integers x and y.
       # Note that f(x, y) is increasing with respect to both x and y.
       # i.e. f(x, y) < f(x + 1, y), f(x, y) < f(x, y + 1)
       def f(self, x, y):

"""


class Solution:
    def findSolution(self, customfunction, z):
        cf = customfunction
        res = []
        if cf.f(1000, 1000) < z or cf.f(1, 1) > z:
            return res
        x, y = 1, z

        while y >= 1 and x <= z:
            if cf.f(x, y) < z:
                x += 1
            elif cf.f(x, y) > z:
                y -= 1
            else:
                res.append([x, y])
                x += 1
                y -= 1

        return res

        # time O(z)
        # space O(1)


# -----------------------------------------------------------------------
"""
557. Reverse Words in a String III

Given a string, you need to reverse the order of characters in each word within a sentence while still preserving 
whitespace and initial word order.

Example 1:
Input: "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
Note: In the string, each word is separated by single space and there will not be any extra space in the string.
"""


class Solution:
    def reverseWords(self, s):
        words = s.split()

        for i in range(len(words)):
            words[i] = words[i][::-1]

        return ' '.join(words)

        # time O(n * m) -> m the length of the longest word
        # space O(n)


# -----------------------------------------------------------------------
"""
897. Increasing Order Search Tree

Given a binary search tree, rearrange the tree in in-order so that the leftmost node in the tree is now the root of 
the tree, and every node has no left child and only 1 right child.

Example 1:
Input: [5,3,6,2,4,null,8,1,null,null,null,7,9]

       5
      / \
    3    6
   / \    \
  2   4    8
 /        / \ 
1        7   9

Output: [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]

 1
  \
   2
    \
     3
      \
       4
        \
         5
          \
           6
            \
             7
              \
               8
                \
                 9 
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.dp = TreeNode('DUMMY')

    def increasingBST(self, root):
        ans = self.dp
        inorder = self.inOrder(root)
        return ans.right

        # time O(n)
        # space O(h) -> the height of the tree

    def inOrder(self, node):
        if not node:
            return

        left = self.inOrder(node.left)
        node.left = None
        self.dp.right = node
        self.dp = node
        right = self.inOrder(node.right)


# -----------------------------------------------------------------------
"""
200. Number of Islands

Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. 
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input:
11110
11010
11000
00000

Output: 1
"""


class Solution:
    def numIslands(self, grid):
        counter = 0

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':
                    self.islandVisit(grid, row, col)
                    counter += 1
        return counter

        # time O(n * m)
        # space O(1)

    def islandVisit(self, grid, row, col):

        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != '1':
            return

        grid[row][col] = '-1'

        self.islandVisit(grid, row, col + 1)
        self.islandVisit(grid, row, col - 1)
        self.islandVisit(grid, row + 1, col)
        self.islandVisit(grid, row - 1, col)


# -----------------------------------------------------------------------
"""
75. Sort Colors

Given an array with n objects colored red, white or blue, sort them in-place so that objects of 
the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:

Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
Follow up:

A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
Could you come up with a one-pass algorithm using only constant space?
"""


class Solution:
    def sortColors(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        #         counter = [0] * 3

        #         for num in nums:
        #             counter[num] += 1

        #         k = 0
        #         for num in range(len(counter)):
        #             while counter[num] > 0:
        #                 nums[k] = num
        #                 k += 1
        #                 counter[num] -= 1
        # time O(n)
        # space O(3)

        start, end, currIdx = 0, len(nums) - 1, 0

        while start < end and currIdx <= end:
            if nums[currIdx] == 0:
                nums[currIdx], nums[start] = nums[start], nums[currIdx]
                start += 1
                currIdx += 1
            elif nums[currIdx] == 2:
                nums[currIdx], nums[end] = nums[end], nums[currIdx]
                end -= 1
            else:
                currIdx += 1

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1403. Minimum Subsequence in Non-Increasing Order

Given the array nums, obtain a subsequence of the array whose sum of elements is strictly greater than the sum of 
the non included elements in such subsequence. 

If there are multiple solutions, return the subsequence with minimum size and if there still exist multiple solutions, 
return the subsequence with the maximum total sum of all its elements. 
A subsequence of an array can be obtained by erasing some (possibly zero) elements from the array. 

Note that the solution with the given constraints is guaranteed to be unique. 
Also return the answer sorted in non-increasing order.

 

Example 1:

Input: nums = [4,3,10,9,8]
Output: [10,9] 
Explanation: The subsequences [10,9] and [10,8] are minimal such that the sum of their elements is strictly greater than the sum of elements not included, however, the subsequence [10,9] has the maximum total sum of its elements. 
"""


class Solution:
    def minSubsequence(self, nums):
        if len(nums) == 1:
            return nums

        total = sum(nums)

        nums.sort()
        res = []
        currSum = 0
        for i in range(len(nums) - 1, -1, -1):
            total -= nums[i]
            currSum += nums[i]
            res.append(nums[i])
            if currSum > total:
                break

        return res

        # time O(n * log(n))
        # space O(n)


# -----------------------------------------------------------------------
"""
559. Maximum Depth of N-ary Tree

Given a n-ary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Nary-Tree input serialization is represented in their level order traversal, each group of children is 
separated by the null value (See examples).

 

Example 1:
Input: root = [1,null,3,2,4,null,5,6]
Output: 3
"""

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


class Solution:
    def maxDepth(self, root):

        # BFS
        if not root:
            return 0

        q = [root]
        maxD = 0

        while len(q) > 0:
            l = len(q)
            maxD += 1
            while l > 0:
                curr = q.pop(0)
                for c in curr.children:
                    if c:
                        q.append(c)
                l -= 1

        return maxD

        # time O(n)
        # space O(w) -> max layer width

        # DFS
        # if not root:
        #     return 0
        #
        # maxD = 0
        # for c in root.children:
        #     maxD = max(maxD, self.maxDepth(c))
        #
        # return maxD + 1

        # time O(n)
        # space O(h) -> max height of the tree


# -----------------------------------------------------------------------
"""
965. Univalued Binary Tree

A binary tree is univalued if every node in the tree has the same value.

Return true if and only if the given tree is univalued.

Example 1:

Input: [1,1,1,1,1,null,1]
Output: true

"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isUnivalTree(self, root):

        seen = set()

        return self.recurIsUniv(root, seen)

    def recurIsUniv(self, node, seen):

        if not node:
            return True

        seen.add(node.val)

        if len(seen) > 1:
            return False

        left = self.recurIsUniv(node.left, seen)

        if not left:
            return False

        right = self.recurIsUniv(node.right, seen)

        return right

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1122. Relative Sort Array

Given two arrays arr1 and arr2, the elements of arr2 are distinct, and all elements in arr2 are also in arr1.

Sort the elements of arr1 such that the relative ordering of items in arr1 are the same as in arr2.  
Elements that don't appear in arr2 should be placed at the end of arr1 in ascending order.

 

Example 1:

Input: arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
Output: [2,2,2,1,4,3,3,9,6,7,19]
"""


class Solution:
    def relativeSortArray(self, arr1, arr2):

        count1 = collections.Counter(arr1)
        arr1 = []
        for num in arr2:
            count = count1[num]
            self.addToArr(arr1, num, count)
            del count1[num]

        for num in sorted(count1):
            count = count1[num]
            self.addToArr(arr1, num, count)

        return arr1

        # time O(n + m)
        # space O(n)

    def addToArr(self, arr, num, count):
        arr.extend([num] * count)


# -----------------------------------------------------------------------
"""
1160. Find Words That Can Be Formed by Characters

You are given an array of strings words and a string chars.

A string is good if it can be formed by characters from chars (each character can only be used once).

Return the sum of lengths of all good strings in words.

 

Example 1:

Input: words = ["cat","bt","hat","tree"], chars = "atach"
Output: 6
Explanation: 
The strings that can be formed are "cat" and "hat" so the answer is 3 + 3 = 6.
"""


class Solution:
    def countCharacters(self, words, chars):

        charsCount = collections.Counter(chars)
        length = 0
        for word in words:
            if self.isValid(word, charsCount):
                length += len(word)

        return length

        # time O(n * l) -> l : the longest word

    def isValid(self, word, charsCount):
        wordCount = collections.Counter(word)
        for char in wordCount:
            if char not in charsCount or wordCount[char] > charsCount[char]:
                return False

        return True


# -----------------------------------------------------------------------
"""
1047. Remove All Adjacent Duplicates In String

Given a string S of lowercase letters, a duplicate removal consists of choosing two adjacent and equal letters, and removing them.

We repeatedly make duplicate removals on S until we no longer can.

Return the final string after all such duplicate removals have been made.  It is guaranteed the answer is unique.

 

Example 1:

Input: "abbaca"
Output: "ca"
Explanation: 
For example, in "abbaca" we could remove "bb" since the letters are adjacent and equal, 
and this is the only possible move.  The result of this move is that the string is 
"aaca", of which only "aa" is possible, so the final string is "ca".
"""


class Solution:
    def removeDuplicates(self, S: str) -> str:

        res = []
        for i in range(len(S)):
            if len(res) > 0 and S[i] == res[-1]:
                res.pop()
            else:
                res.append(S[i])

        return ''.join(res)

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
876. Middle of the Linked List

Given a non-empty, singly linked list with head node head, return a middle node of linked list.

If there are two middle nodes, return the second middle node.

 

Example 1:

Input: [1,2,3,4,5]
Output: Node 3 from this list (Serialization: [3,4,5])
The returned node has value 3.  (The judge's serialization of this node is [3,4,5]).
Note that we returned a ListNode object ans, such that:
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, and ans.next.next.next = NULL.
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head):
        if not head:
            return head

        slow, fast = head, head.next

        while fast:
            fast = fast.next
            if fast:
                fast = fast.next
            slow = slow.next

        return slow

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
1200. Minimum Absolute Difference

Given an array of distinct integers arr, find all pairs of elements with the minimum absolute difference of any two elements. 

Return a list of pairs in ascending order(with respect to pairs), each pair [a, b] follows

a, b are from arr
a < b
b - a equals to the minimum absolute difference of any two elements in arr
 

Example 1:

Input: arr = [4,2,1,3]
Output: [[1,2],[2,3],[3,4]]
Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.
"""

import sys


class Solution:
    def minimumAbsDifference(self, arr):

        arr.sort()
        res = []
        minDiff = sys.maxsize

        for i in range(1, len(arr)):
            if arr[i] - arr[i - 1] < minDiff:
                minDiff = arr[i] - arr[i - 1]

        for i in range(1, len(arr)):
            if arr[i] - arr[i - 1] == minDiff:
                res.append([arr[i - 1], arr[i]])

        return res

        # time O(n * log(n))
        # space O(n)




# -----------------------------------------------------------------------
"""
215. Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, 
not the kth distinct element.

Example 1:

Input: [3,2,1,5,6,4] and k = 2
Output: 5
Example 2:

Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
"""


class Solution:
    def findKthLargest(self, nums, k):
        minHeap = []

        for num in nums:
            if len(minHeap) < k:
                heapq.heappush(minHeap, num)
            else:
                last = heapq.heappop(minHeap)
                if num > last:
                    last = num
                heapq.heappush(minHeap, last)

        return heapq.heappop(minHeap)









