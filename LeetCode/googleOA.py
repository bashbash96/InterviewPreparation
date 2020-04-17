import sys
import collections
import heapq

# -----------------------------------------------------------------------
"""
Given an Array A, find the minimum amplitude you can get after changing up to 3 elements. 
Amplitude is the range of the array (basically difference between largest and smallest element).

Example 1:

Input: [-1, 3, -1, 8, 5 4]
Output: 2
Explanation: we can change -1, -1, 8 to 3, 4 or 5

Example 2:

Input: [10, 10, 3, 4, 10]
Output: 0
Explanation: change 3 and 4 to 10
"""


def minAplitude(arr):
    minHeap, maxHeap = [], []

    for num in arr:
        addToMaxHeap(maxHeap, num)
        addToMinHeap(minHeap, num)

    minHeap.sort()
    maxHeap.sort()
    res = maxHeap[len(maxHeap) - 1] - minHeap[0]
    res = min(res, maxHeap[0] - minHeap[0])  # replace all 3 max
    res = min(res, maxHeap[1] - minHeap[1])  # replace 2 from max and 1 from min
    res = min(res, maxHeap[2] - minHeap[2])  # replace 1 from max and 2 from min
    res = min(res, maxHeap[2] - minHeap[2])  # replace all 3 min

    return res

    # time O(n)
    # space O(1)


def addToMinHeap(minHeap, num):
    if len(minHeap) < 4:
        heapq.heappush(minHeap, num)
    else:
        last = heapq.heappop(minHeap)
        if num < last:
            last = num
        heapq.heappush(minHeap, last)


def addToMaxHeap(maxHeap, num):
    if len(maxHeap) < 4:
        heapq.heappush(maxHeap, num)
    else:
        last = heapq.heappop(maxHeap)
        if num > last:
            last = num
        heapq.heappush(maxHeap, last)


# -----------------------------------------------------------------------
"""
Given a string S, we can split S into 2 strings: S1 and S2. Return the number of ways S can be split 
such that the number of unique characters between S1 and S2 are the same.

Example 1:

Input: "aaaa"
Output: 3
Explanation: we can get a - aaa, aa - aa, aaa- a

Example 2:

Input: "bac"
Output: 0

"""


def numOfSplit(S):
    rightCount, leftCount = collections.Counter(S), collections.defaultdict(int)

    res = 0
    for i in range(len(S) - 1):
        leftCount[S[i]] += 1
        rightCount[S[i]] -= 1
        if rightCount[S[i]] == 0:
            del rightCount[S[i]]
        if isEqual(leftCount, rightCount):
            res += 1
    return res

    # time O(n)
    # space O(n)


def isEqual(left, right):
    if len(left) != len(right):
        return False

    for key in left:
        if key not in right:
            return False

    return True


# -----------------------------------------------------------------------
"""
one string is strictly smaller than another when the frequency of occurrence of the smallest
character in the string is less than the frequency of the occurrence of the 
smallest character in the comparison string

for example string 'abcd' is smaller than string 'aaa' because the smallest character in 'abcd' is 'a'
with frequency of 1, and the smallest character in 'aaa' is also 'a' but with frequency of 3.

write a function that given a string A (which contains M strings delimited by ',') and string B 
(which contains N string delimted by ',') returns an array of c of N integers. for 0 <= j < N, values 
of C[j] specify the number of strings in A which are strictly smaller than the comparison j-th string in B

Example:

A = 'abcd,aabc,bd'
B = 'aaa,aa'

return [3,2]

Assume that:
- 1 <= N,m <= 10000
- 1 <= length of strings in A or B <= 10
- all character are lowercases alphabet
"""


def compareStrings(A, B):
    wordsA = A.split(',')
    wordsB = B.split(',')
    freqs = [0] * 11

    for w in wordsA:
        minChar = min(w)
        freqs[w.count(minChar)] += 1

    for i in range(1, len(freqs)):
        freqs[i] += freqs[i - 1]

    for i in range(len(wordsB)):
        currWord = wordsB[i]
        minChar = min(currWord)
        wordsB[i] = freqs[currWord.count(minChar) - 1]

    return wordsB

    # time O(n + m)
    # space O(n + m)


# -----------------------------------------------------------------------

"""
Array X is greater than array Y if the first non-matching element in both arrays has a greater value in X than in Y

for example: 

X = [1,2,4,3,5]
Y = [1,2,3,4,5]
X is greater than Y because the first element that does not match is larger in X (X[2] and Y[2], X[2] > Y[2])

A contiguous subarray if defined by an interval of the indices. in other words a contiguous subarray is a subarray 
which has consecutive indexes.

write a function that, given a zero-indexed array A consisting of N integers and an integer K, returns the largest contiguous
sub arrays of length K from all the contiguous subarrays of length K

for example, given array A and k = 4 such that:

A = [1,4,3,2,5]

the function should return [4,3,2,5] because there are two subarrays of size 4:

[1,4,3,2]
[4,3,2,5]

and the largest subarray is [4,3,2,5]

assume that:
1 <= k <= N <= 100
1 <= A[j] <= 1000
"""


def largestKSub(arr, k):
    if k > len(arr) or k <= 0:
        return []

    res = arr[:k]
    for i in range(1, len(arr) - k + 1):
        curr = arr[i:i + k]
        if isLarger(curr, res):
            res = [val for val in curr]

    return res

    # time O((n-k)*k)
    # space O(k)


def isLarger(first, second):
    p = 0
    while p < len(first) and first[p] == second[p]:
        p += 1

    return p < len(first) and first[p] > second[p]


# -----------------------------------------------------------------------
"""
You are given a string that represents time in the format hh:mm. Some of the digits are blank
(represented by ?). Fill in ? such that the time represented by this string is the maximum possible. 
Maximum time: 23:59, minimum time: 00:00. You can assume that input string is always valid.

Example 1:

Input: "?4:5?"
Output: "14:59"

Example 2:

Input: "23:5?"
Output: "23:59"

Example 3:

Input: "2?:22"
Output: "23:22"

Example 4:

Input: "0?:??"
Output: "09:59"

Example 5:

Input: "??:??"
Output: "23:59"

"""


def maxTime(time):
    time = [c for c in time]
    time[4] = time[4] if time[4] != '?' else '9'
    time[3] = time[3] if time[3] != '?' else '5'

    if time[1] == '?':
        if time[0] == '?' or time[0] == '2':
            time[1] = '3'
        else:
            time[1] = '9'

    if time[0] == '?':
        if time[1] <= '3':
            time[0] = '2'
        else:
            time[0] = '1'

    return ''.join(time)

    # time O(5)
    # space O(5)


# -----------------------------------------------------------------------
"""
There are some processes that need to be executed. Amount of a load that process causes on a server that runs it, 
is being represented by a single integer. Total load caused on a server is the sum of the loads of all the 
processes that run on that server. You have at your disposal two servers, on which mentioned processes can be run. 
Your goal is to distribute given processes between those two servers in the way that, 
absolute difference of their loads will be minimized.

Given an array of n integers, of which represents loads caused by successive processes, return the minimum absolute 
difference of server loads.

Example 1:

Input: [1, 2, 3, 4, 5]
Output: 1
Explanation:
We can distribute the processes with loads [1, 2, 4] to the first server and [3, 5] to the second one,
so that their total loads will be 7 and 8, respectively, and the difference of their loads will be equal to 1.
"""


def minAbs(nums):
    s = sum(nums) // 2
    n = len(nums)
    memo = [[0 for j in range(s + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, s + 1):
            if nums[i - 1] <= j:
                memo[i][j] = max(memo[i - 1][j], nums[i - 1] + memo[i - 1][j - nums[i - 1]])
            else:
                memo[i][j] = memo[i - 1][j]
    '''return second server loads - first server loads'''
    return sum(nums) - (2 * memo[n][s])


# -----------------------------------------------------------------------
"""
Given a hotel which has 10 floors [0-9] and each floor has 26 rooms [A-Z]. You are given a sequence of rooms, 
where + suggests room is booked, - room is freed. You have to find which room is booked maximum number of times.

You may assume that the list describe a correct sequence of bookings in chronological order; that is, 
only free rooms can be booked and only booked rooms can be freeed. All rooms are initially free. 
Note that this does not mean that all rooms have to be free at the end. In case, 
2 rooms have been booked the same number of times, return the lexographically smaller room.

You may assume:

N (length of input) is an integer within the range [1, 600]
each element of array A is a string consisting of three characters: "+" or "-"; a digit "0"-"9"; 
and uppercase English letter "A" - "Z"
the sequence is correct. That is every booked room was previously free and every freed room was previously booked.

Example:
Input: ["+1A", "+3E", "-1A", "+4F", "+1A", "-3E"]
Output: "1A"
Explanation: 1A as it has been booked 2 times.
"""


def maxBookedRoom(rooms):
    if len(rooms) == 0:
        return None

    count = collections.defaultdict(int)
    currMax = ()
    for room in rooms:
        if room[0] == '+':
            room = room[1:]
            count[room] += 1
            if currMax == () or count[room] > currMax[1]:
                currMax = (room, count[room])
            elif count[room] == currMax[1]:
                if room < currMax[0]:
                    currMax = (room, count[room])

    return currMax[0]


# -----------------------------------------------------------------------
"""
You and your friend gardeners, and you take care of your plants. The plants are planted in a row, and each of them needs
a specific amount of water. You are about to water them using watering cans. To avoid mistakes like applying too much water,
or not watering a plant at all, you have decided to:
- water the plants in the order in which they appear: you will water from left to rught, and your friend will water from right
to left
- water each plant if you have sufficient water for it, otherwise refill your watering can.
- water each plant in one go, i.e. without taking a break to refill the watering can in the middle of watering a single plant.
This means that you may sometimes have to refill your watering can before or after watering a plant, even though it's not
completely empty.

you start with watering the first plant and your friend start with watering the last plant. You and your friend are watering
the plants simultaneously (when you are watering the first plant, your friend is watering the last one and so on).
That means that you will meet in the middle of the row of plants if there is an unwatered plant there, and you and your
friend together hav enough water for it, you can water it without refilling your watering cans. otherwise only on of you should refill

at the beginning you both start with empty watering cans. How many times will you and yourfriend need to refill your watering
cans in order to water all the plants in the row?

write a function:
def Solution(plants, capacity1, capacity2)

that given an array of plants of N integers (for the amount of water needed by each plant), and variables capacity1 and 
capacity2 (for the capacity of your watering can and your friend's). return the number of times you and your friend will
have to refill your watering cans to water all the plants.

for example, given plants = [2,4,5,1,2], capacity1 = 5 and capacity2 = 7, the function should return 3.

"""


def numOfRefills(plants, capacity1, capacity2):
    if len(plants) <= 1:
        return 0
    left, right, leftCan, rightCan, counter = 0, len(plants) - 1, capacity1, capacity2, 2
    while left <= right:
        if left == right:
            if leftCan + rightCan < plants[left]:
                counter += 1
            break
        counter, leftCan = waterOrRefill(plants, left, leftCan, capacity1, counter)
        counter, rightCan = waterOrRefill(plants, right, rightCan, capacity2, counter)
        left += 1
        right -= 1
    return counter

    # time O(n)
    # space O(1)


def waterOrRefill(plants, idx, can, capacity, counter):
    if can < plants[idx]:
        counter += 1
        can = capacity
    can -= plants[idx]
    return counter, can


# -----------------------------------------------------------------------
"""
1007. Minimum Domino Rotations For Equal Row

In a row of dominoes, A[i] and B[i] represent the top and bottom halves of the i-th domino.  
(A domino is a tile with two numbers from 1 to 6 - one on each half of the tile.)

We may rotate the i-th domino, so that A[i] and B[i] swap values.

Return the minimum number of rotations so that all the values in A are the same, or all the values in B are the same.

If it cannot be done, return -1.

Example 1:

Input: A = [2,1,2,4,2,2], B = [5,2,6,2,3,2]
Output: 2
Explanation: 
The first figure represents the dominoes as given by A and B: before we do any rotations.
If we rotate the second and fourth dominoes, we can make every value in the top row equal to 2, 
as indicated by the second figure.
"""


class Solution:
    def minDominoRotations(self, A, B):
        maxA = self.getMax(A)
        maxB = self.getMax(B)
        rotationsA = self.getNumOfRot(A, B, maxA)
        rotationsB = self.getNumOfRot(B, A, maxB)

        if rotationsA == -1 and rotationsB == -1:
            return -1

        if rotationsA == -1:
            return rotationsB

        if rotationsB == -1:
            return rotationsA

        return min(rotationsA, rotationsB)

        # time O(n)
        # space O(1)

    def getNumOfRot(self, first, second, maxVal):
        counter = 0
        for i in range(len(first)):
            if first[i] != maxVal:
                if second[i] != maxVal:
                    return -1
                counter += 1

        return counter

    def getMax(self, arr):
        count = collections.Counter(arr)
        maxVal = max(count.values())
        for key in count:
            if count[key] == maxVal:
                maxVal = key
                break
        return maxVal


# -----------------------------------------------------------------------
"""
Imagine you have a special keyboard with all keys in a single row. The layout of characters on a keyboard is denoted by a string keyboard of length 26. 
Initially your finger is at index 0. To type a character, you have to move your finger to the index of the desired character. 
The time taken to move your finger from index i to index j is abs(j - i).

Given a string keyboard that describe the keyboard layout and a string text, return an integer denoting the time taken to type string text.

Example 1:

Input: keyboard = "abcdefghijklmnopqrstuvwxy", text = "cba" 
Output: 4
Explanation:
Initially your finger is at index 0. First you have to type 'c'. The time taken to type 'c' will be abs(2 - 0) = 2 because character 'c' is at index 2.
The second character is 'b' and your finger is now at index 2. The time taken to type 'b' will be abs(1 - 2) = 1 because character 'b' is at index 1.
The third character is 'a' and your finger is now at index 1. The time taken to type 'a' will be abs(0 - 1) = 1 because character 'a' is at index 0.
The total time will therefore be 2 + 1 + 1 = 4.
Constraints:

length of keyboard will be equal to 26 and all the lowercase letters will occur exactly once;
the length of text is within the range [1..100,000];
string text contains only lowercase letters [a-z].

"""


def timeTaken(keyboard, text):
    map = {}
    for i in range(len(keyboard)):
        map[keyboard[i]] = i

    time = 0
    prevChar = 'a'
    for currChar in text:
        time += abs(map[currChar] - map[prevChar])
        prevChar = currChar

    return time

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
1161. Maximum Level Sum of a Binary Tree

Given the root of a binary tree, the level of its root is 1, the level of its children is 2, and so on.

Return the smallest level X such that the sum of all the values of nodes at level X is maximal.

Example 1:

Input: [1,7,0,7,-8,null,null]
Output: 2
Explanation: 
Level 1 sum = 1.
Level 2 sum = 7 + 0 = 7.
Level 3 sum = 7 + -8 = -1.
So we return the level with the maximum sum which is level 2.
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxLevelSum(self, root):
        if not root:
            return 0

        maxLevel = (1, root.val)
        q = [root]
        level = 1
        while len(q) > 0:
            levelLength = len(q)
            currSum = 0
            while levelLength > 0:
                currNode = q.pop(0)
                currSum += currNode.val
                if currNode.left:
                    q.append(currNode.left)
                if currNode.right:
                    q.append(currNode.right)
                levelLength -= 1

            if currSum > maxLevel[1]:
                maxLevel = (level, currSum)

            level += 1

        return maxLevel[0]

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
There are n guests who are invited to a party. The k-th guest will attend the party at time S[k] and leave the party at time E[k].

Given an integer array S and an integer array E, both of length n, return an integer denoting 
the minimum number of chairs you need such that everyone attending the party can sit down.

Example:

Input: S = [1, 2, 6, 5, 3], E = [5, 5, 7, 6, 8]
Output: 3
Explanation:
There are five guests attending the party. 
The 1st guest will arrive at time 1. We need one chair at time 1.
The 2nd guest will arrive at time 2. There are now two guests at the party, so we need two chairs at time 2.
The 5th guest will arrive at time 3. There are now three guests at the party, so we need three chairs at time 3.
The 4th guest will arrive at time 5 and, at the same moment, the 1st and 2nd guests will leave the party.
There are now two (the 4th and 5th) guests at the party, so we need two chairs at time 5.
The 3rd guest will arrive at time 6, and the 4th guest will leave the party at the same time.
There are now two (the 3rd and 5th) guests at the party, so we need two chairs at time 6. 
So we need at least 3 chairs

"""


def minChairs(S, E):
    times = mergeArrs(S, E)
    counter, maxChairs = 0, 0
    for time in times:
        if time[1] == 1:
            counter += 1
        else:
            counter -= 1
        maxChairs = max(maxChairs, counter)
    return maxChairs

    # time O((n+m) * log(n+m))
    # space O(n+m)


def mergeArrs(S, E):
    res = []
    for num in S:
        res.append((num, 1))  # 1 for arrival
    for num in E:
        res.append((num, 0))  # 0 for leaving
    res = sorted(res, key=lambda pair: (pair[0], pair[1]))
    return res


# -----------------------------------------------------------------------
"""
973. K Closest Points to Origin

We have a list of points on the plane.  Find the K closest points to the origin (0, 0).

(Here, the distance between two points on a plane is the Euclidean distance.)

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)



Example 1:

Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]
Explanation: 
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].
"""


class Solution:
    def kClosest(self, points, K):
        if K == len(points):
            return points

        maxHeap = []

        for point in points:
            if len(maxHeap) < K:
                heapq.heappush(maxHeap, (self.getDistance(point) * -1, point))
            else:
                curr = heapq.heappop(maxHeap)
                distance = self.getDistance(point)
                if distance < curr[0] * -1:
                    curr = (distance * -1, point)
                heapq.heappush(maxHeap, curr)

        return [pair[1] for pair in maxHeap]

        # time O(n * log(k))
        # space O(k)

    def getDistance(self, point):
        x = pow(point[0], 2)
        y = pow(point[1], 2)
        return pow(x + y, 0.5)


# -----------------------------------------------------------------------
"""
975. Odd Even Jump

You are given an integer array A.  From some starting index, you can make a series of jumps.  The (1st, 3rd, 5th, ...) jumps in the series are called odd numbered jumps, and the (2nd, 4th, 6th, ...) jumps in the series are called even numbered jumps.

You may from index i jump forward to index j (with i < j) in the following way:

During odd numbered jumps (ie. jumps 1, 3, 5, ...), you jump to the index j such that A[i] <= A[j] and A[j] is the 
smallest possible value.  If there are multiple such indexes j, you can only jump to the smallest such index j.
During even numbered jumps (ie. jumps 2, 4, 6, ...), you jump to the index j such that A[i] >= A[j] and A[j] is the 
largest possible value.  If there are multiple such indexes j, you can only jump to the smallest such index j.
(It may be the case that for some index i, there are no legal jumps.)
A starting index is good if, starting from that index, you can reach the end of the array (index A.length - 1) by 
jumping some number of times (possibly 0 or more than once.)

Return the number of good starting indexes.

 

Example 1:

Input: [10,13,12,14,15]
Output: 2
Explanation: 
From starting index i = 0, we can jump to i = 2 (since A[2] is the smallest among A[1], A[2], A[3], A[4] that is 
greater or equal to A[0]), then we can't jump any more.
From starting index i = 1 and i = 2, we can jump to i = 3, then we can't jump any more.
From starting index i = 3, we can jump to i = 4, so we've reached the end.
From starting index i = 4, we've reached the end already.
In total, there are 2 different starting indexes (i = 3, i = 4) where we can reach the end with some number of jumps.
"""


class Solution(object):
    def oddEvenJumps(self, A):
        N = len(A)

        def make(B):
            ans = [None] * N
            stack = []  # invariant: stack is decreasing
            for i in B:
                while stack and i > stack[-1]:
                    ans[stack.pop()] = i
                stack.append(i)
            return ans

        B = sorted(range(N), key=lambda i: A[i])
        oddnext = make(B)
        B.sort(key=lambda i: -A[i])
        evennext = make(B)

        odd = [False] * N
        even = [False] * N
        odd[N - 1] = even[N - 1] = True

        for i in range(N - 2, -1, -1):
            if oddnext[i] is not None:
                odd[i] = even[oddnext[i]]
            if evennext[i] is not None:
                even[i] = odd[evennext[i]]

        return sum(odd)


# -----------------------------------------------------------------------
"""
482. License Key Formatting

You are given a license key represented as a string S which consists only alphanumeric character and dashes. 
The string is separated into N+1 groups by N dashes.

Given a number K, we would want to reformat the strings such that each group contains exactly K characters, 
except for the first group which could be shorter than K, but still must contain at least one character. Furthermore, 
there must be a dash inserted between two groups and all lowercase letters should be converted to uppercase.

Given a non-empty string S and a number K, format the string according to the rules described above.

Example 1:
Input: S = "5F3Z-2e-9-w", K = 4

Output: "5F3Z-2E9W"

Explanation: The string S has been split into two parts, each part has 4 characters.
Note that the two extra dashes are not needed and can be removed.
"""


class Solution:
    def licenseKeyFormatting(self, S, K):
        res = []
        alphanum = self.getAlphaNumeric(S)
        curr = []
        for i in range(len(alphanum) - 1, -1, -1):
            curr.append(alphanum[i])
            if len(curr) == K:
                res.append(reversed(curr))
                curr = []
        if curr:
            res.append(reversed(curr))

        res.reverse()
        temp = ''
        for key in res:
            for val in key:
                temp += val
            temp += '-'

        return temp[:-1]

    def getAlphaNumeric(self, strs):
        alphanum = []
        for c in strs:
            if c.isalnum():
                if c.isalpha():
                    alphanum.append(c.upper())
                else:
                    alphanum.append(c)
        return alphanum


# -----------------------------------------------------------------------
"""
929. Unique Email Addresses

Every email consists of a local name and a domain name, separated by the @ sign.

For example, in alice@leetcode.com, alice is the local name, and leetcode.com is the domain name.

Besides lowercase letters, these emails may contain '.'s or '+'s.

If you add periods ('.') between some characters in the local name part of an email address, mail sent there 
will be forwarded to the same address without dots in the local name.  For example, 
"alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.  
(Note that this rule does not apply for domain names.)

If you add a plus ('+') in the local name, everything after the first plus sign will be ignored. 
This allows certain emails to be filtered, for example m.y+name@email.com will be forwarded to my@email.com.  
(Again, this rule does not apply for domain names.)

It is possible to use both of these rules at the same time.

Given a list of emails, we send one email to each address in the list.  
How many different addresses actually receive mails? 



Example 1:

Input: ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
Output: 2
Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails
"""


class Solution:
    def numUniqueEmails(self, emails):

        uniqueMails = set()

        for mail in emails:
            name, domain = mail.split('@')
            name = name.replace('.', '')
            if '+' in name:
                name = name[:name.index('+')]
            newMail = name + '@' + domain
            uniqueMails.add(newMail)

        return len(uniqueMails)

        # time O(n * l) -> l : longest mail
        # space O(n)


# -----------------------------------------------------------------------
"""
904. Fruit Into Baskets

In a row of trees, the i-th tree produces fruit with type tree[i].

You start at any tree of your choice, then repeatedly perform the following steps:

Add one piece of fruit from this tree to your baskets.  If you cannot, stop.
Move to the next tree to the right of the current tree.  If there is no tree to the right, stop.
Note that you do not have any choice after the initial choice of starting tree: you must perform step 1, 
then step 2, then back to step 1, then step 2, and so on until you stop.

You have two baskets, and each basket can carry any quantity of fruit, but you want each basket to 
only carry one type of fruit each.

What is the total amount of fruit you can collect with this procedure?



Example 1:

Input: [1,2,1]
Output: 3
Explanation: We can collect [1,2,1].
"""


class Solution:
    def totalFruit(self, tree):
        idx, maxLen, map = 0, 0, collections.defaultdict(int)

        for i in range(len(tree)):
            map[tree[i]] += 1
            while len(map) >= 3:
                map[tree[idx]] -= 1
                if map[tree[idx]] == 0:
                    del map[tree[idx]]
                idx += 1
            maxLen = max(maxLen, i - idx + 1)

        return maxLen

        # time O(n)
        # space O(n)


# -----------------------------------------------------------------------
"""
Given an array of roses. roses[i] means rose i will bloom on day roses[i]. Also given an int k, which is the minimum 
number of adjacent bloom roses required for a bouquet, and an int n, which is the number of bouquets we need. 
Return the earliest day that we can get n bouquets of roses.

Example:
Input: roses = [1, 2, 4, 9, 3, 4, 1], k = 2, n = 2
Output: 4
Explanation:
day 1: [b, n, n, n, n, n, b]
The first and the last rose bloom.

day 2: [b, b, n, n, n, n, b]
The second rose blooms. Here the first two bloom roses make a bouquet.

day 3: [b, b, n, n, b, n, b]

day 4: [b, b, b, n, b, b, b]
Here the last three bloom roses make a bouquet, meeting the required n = 2 bouquets of bloom roses. So return day 4.
"""


def earliestDay(roses, k, n):
    start, end = 1, max(roses)
    day = end

    while start <= end:
        mid = (start + end) // 2
        if isEnoughBouquets(roses, mid, k, n):
            day = min(day, mid)
            end = mid - 1
        else:
            start = mid + 1

    return day

    # time O(n * log(n))
    # space O(1)


def isEnoughBouquets(roses, day, k, n):
    currCount, res = 0, 0

    for d in roses:
        if d <= day:
            currCount += 1
            if currCount == k:
                res += 1
                currCount = 0
        else:
            currCount = 0
        if res >= n:
            return True

    return False


# -----------------------------------------------------------------------

"""
Given a NxN matrix. Fill the integers from 1 to n*n to this matrix that makes the sum of each row, 
each column and the two diagonals equal.

Example 1:

Input: n = 2
Output: null
Explanation: We need to fill [1, 2, 3, 4] into a 2x2 matrix, which is not possible so return null.
"""


def generateSquare(n):
    magicSquare = [[0 for x in range(n)]
                   for y in range(n)]

    # initialize position of 1
    i = n / 2
    j = n - 1

    num = 1
    while num <= (n * n):
        if i == -1 and j == n:
            j = n - 2
            i = 0
        else:
            if j == n:
                j = 0
            if i < 0:
                i = n - 1

        if magicSquare[int(i)][int(j)]:
            j = j - 2
            i = i + 1
            continue
        else:
            magicSquare[int(i)][int(j)] = num
            num = num + 1
        j = j + 1
        i = i - 1

    return magicSquare

    # time O(n^2)
    # space O(n^2)


# -----------------------------------------------------------------------
"""
Given an int array nums of length n. Split it into strictly decreasing subsequences. 
Output the min number of subsequences you can get by splitting.

Example 1:

Input: [5, 2, 4, 3, 1, 6]
Output: 3
Explanation:
You can split this array into: [5, 2, 1], [4, 3], [6]. And there are 3 subsequences you get.
Or you can split it into [5, 4, 3], [2, 1], [6]. Also 3 subsequences.
But [5, 4, 3, 2, 1], [6] is not legal because [5, 4, 3, 2, 1] is not a subsuquence of the original array.

Example 2:

Input: [2, 9, 12, 13, 4, 7, 6, 5, 10]
Output: 4
Explanation: [2], [9, 4], [12, 10], [13, 7, 6, 5]

Example 3:

Input: [1, 1, 1]
Output: 3
Explanation: Because of the strictly descending order you have to split it into 3 subsequences: [1], [1], [1]
"""


def minSub(arr):
    res = 0

    for i in range(len(arr)):
        if arr[i] == sys.maxsize:
            continue
        currNum = arr[i]
        arr[i] = sys.maxsize
        for j in range(i + 1, len(arr)):
            if arr[j] != sys.maxsize and arr[j] < currNum:
                currNum = arr[j]
                arr[j] = sys.maxsize
        res += 1

    return res

    # time O(n^2)
    # space O(1)


# -----------------------------------------------------------------------
"""
The distance between 2 binary strings is the sum of their lengths after removing the common prefix. 
For example: the common prefix of 1011000 and 1011110 is 1011 so the distance is len("000") + len("110") = 3 + 3 = 6.

Given a list of binary strings, pick a pair that gives you maximum distance among all possible pair and return that distance.
"""


class Trie:
    def __init__(self):
        self.root = {}

    def add(self, s):
        curr = self.root
        for c in s:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]

        curr['*'] = True

    def print(self):
        curr = self.root
        self.recurPrint(curr, [])

    def recurPrint(self, node, curr):
        if not node or node == True:
            print(''.join(curr[:-1]))
            return

        for c in node:
            curr.append(c)
            self.recurPrint(node[c], curr)
            curr.pop()


def maxDistance(strings):
    tree = Trie()
    for s in strings:
        tree.add(s)
    curr = tree.root
    return getMaxDistance(curr)[1]

    # time O(n * k) -> n number of strings, k largest string
    # space O( n * k )


def getMaxDistance(node):
    if not node or node == '*' or node == True:
        return 0, 0  # -> (max depth, max distance)

    res = []
    for c in node:
        res.append(getMaxDistance(node[c]))

    currLen = 0
    distance = 0
    prevDistance = 0
    for pair in res:
        currLen = max(currLen, pair[0])
        distance += pair[0]
        prevDistance = max(prevDistance, pair[1])

    if len(res) < 2:
        distance = 0

    return currLen + 1, max(distance, prevDistance)


# -----------------------------------------------------------------------
"""
You are given 2 arrays representing integer locations of stores and houses (each location in this problem 
is one-dementional). For each house, find the store closest to it.
Return an integer array result where result[i] should denote the location of the store closest to the i-th house. 
If many stores are equidistant from a particular house, choose the store with the smallest numerical location. 
Note that there may be multiple stores and houses at the same location.

Example 1:

Input: houses = [5, 10, 17], stores = [1, 5, 20, 11, 16]
Output: [5, 11, 16]
Explanation: 
The closest store to the house at location 5 is the store at the same location.
The closest store to the house at location 10 is the store at the location 11.
The closest store to the house at location 17 is the store at the location 16.
Example 2:

Input: houses = [2, 4, 2], stores = [5, 1, 2, 3]
Output: [2, 3, 2]
Example 3:

Input: houses = [4, 8, 1, 1], stores = [5, 3, 1, 2, 6]
Output: [3, 6, 1, 1]
"""


def closestStore(houses, stores):
    if len(stores) < 1:
        return [sys.maxsize] * len(houses)

    stores = set(stores)
    stores = [val for val in stores]

    for i in range(len(houses)):
        houses[i] = getClosestStore(stores, houses[i])

    return houses

    # time O(n * log(n))
    # space O(1)


def getClosestStore(stores, house):
    start, end = 0, len(stores) - 1
    res = (stores[0], abs(stores[0] - house))

    while start <= end:
        mid = (start + end) // 2
        store = stores[mid]
        distance = abs(house - store)

        if res[1] > distance:
            res = (store, distance)
        elif res[1] == distance:
            if res[0] > store:
                res = (store, distance)
        if store == house:
            return res[0]
        elif house < store:
            end = mid - 1
        else:
            start = mid + 1

    return res[0]


# -----------------------------------------------------------------------
"""
GCD (Greatest Common Divisor) of two positive integers is the largest positive integer that divides both numbers 
without a remainder.
Siblings: Nodes with the same parent are called siblings.
Level of a tree: Level of a tree is the number of edges on the longest path from the root node to a leaf.
You are given nodes of a binary tree of leven n as input.
Caluclate the GCD of each pair of siblings and then find the max & min GCD among them. 
Print the difference of max & min GCD ( max GCD - min GCD)

Note:
Print -1 if input tree is empty i.e level of tree is -1.
Consider those nodes which have a sibling
Print 0 if no such pair of siblings found
Input Format:
The input is in the following format:

The first line takes an integer n as input which represents the level of tree (the root node is at 0 level). 
(if level is equal to -1, means empty tree)
Next n+1 lines contain the nodes in the tree level order. Each i'th line represents the nodes present in the 
binary tree in i'th level.
1st line contains level 0 nodes. (i.e. root node).
2nd line contains nodes for level 1.
3rd line contains nodes for level 2 and so on.
Each node is represented by an integer value. Node value of -1 denotes an empty node(no node present at that place).

Output Format:
A single integer i.e., the difference of max & min GCD (max GCD - min GCD)

Constraints:
-1 <= level of tree <= 20
0 < element at nodes of tree <= 500
"""


def diffGCD():
    minGCD, maxGCD = sys.maxsize, -1 * sys.maxsize

    n = int(input())
    if n == -1:
        return -1

    for i in range(n + 1):
        if i == 0:
            input()  # --- for the root of the tree
            continue
        currLevel = list(map(int, input().split()))
        for j in range(0, pow(2, i), 2):
            first, second = currLevel[j], currLevel[j + 1]

            if first == -1 or second == -1:
                continue
            currGCD = gcd(first, second)

            if currGCD < minGCD:
                minGCD = currGCD

            if currGCD > maxGCD:
                maxGCD = currGCD

    if minGCD == sys.maxsize:
        return 0

    return maxGCD - minGCD

    # time O(n)
    # space O(w) -> max width


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


# -----------------------------------------------------------------------
"""
801. Minimum Swaps To Make Sequences Increasing

We have two integer sequences A and B of the same non-zero length.

We are allowed to swap elements A[i] and B[i].  Note that both elements are in the same index position in their respective sequences.

At the end of some number of swaps, A and B are both strictly increasing.  (A sequence is strictly increasing if and only if A[0] < A[1] < A[2] < ... < A[A.length - 1].)

Given A and B, return the minimum number of swaps to make both sequences strictly increasing.  It is guaranteed that the given input always makes it possible.

Example:
Input: A = [1,3,5,4], B = [1,2,3,7]
Output: 1
Explanation: 
Swap A[3] and B[3].  Then the sequences are:
A = [1, 3, 5, 7] and B = [1, 2, 3, 4]
which are both strictly increasing.

"""


class Solution:
    def minSwap(self, A, B):
        noSwap1, swap1 = 0, 1

        for i in range(1, len(A)):
            noSwap2 = swap2 = float('inf')

            if A[i - 1] < A[i] and B[i - 1] < B[i]:
                noSwap2 = min(noSwap2, noSwap1)
                swap2 = min(swap2, swap1 + 1)
            if A[i - 1] < B[i] and B[i - 1] < A[i]:
                noSwap2 = min(noSwap2, swap1)
                swap2 = min(swap2, noSwap1 + 1)

            noSwap1, swap1 = noSwap2, swap2

        return min(noSwap1, swap1)

        # time O(n)
        # space O(1)


# -----------------------------------------------------------------------
"""
A robot wnats to pick strawberries from a strawberry bush. You are given an integer array arr of length n and an integer num
as input where num is the number of strawberries a robot will pick as maximum.
Each array element in arr represents the number of strawberries present int eah bush and n is the number of bushes. 
A robot cannot pick strawberries from two consecutive bushes and it has to pick the strawberries in such a way that it 
may not exceed the limit proposed i.e., num. Calculate the maximum number of strawberries a robot can pick and print it.

Note: print 0 if it is not possible to pick strawberries within the maximum number of strawberries that can be collected by a robot.

Input Format:
Each test case consist of three lines of input:
- first line consist of an integer i.e., the maximum number of strawberries a robot can pick
- second line consist of an integer i.e., number of bushes, n.
- third line consist of n integers seperated by space i.e., the number of strawberries in each bush.

output Format:
A singl integer i.e., the maximum number of strawberries collected by the robot.

"""


def main():
    target = int(input())
    n = int(input())
    arr = list(map(int, input().split()))


def maxStrawberries(strawberries, target):
    memo = [[None for j in range(target + 1)] for i in range(len(strawberries))]

    return recurMaxStrawberries(strawberries, target, 0, 0, memo)

    # time O(n * s)
    # space O(n * s)


def recurMaxStrawberries(strawberries, target, idx, curr, memo):
    if idx >= len(strawberries):
        return curr

    if not memo[idx][curr]:
        taken = 0
        if strawberries[idx] + curr <= target:
            taken = recurMaxStrawberries(strawberries, target, idx + 2, curr + strawberries[idx], memo)
        notTaken = recurMaxStrawberries(strawberries, target, idx + 1, curr, memo)

        memo[idx][curr] = max(taken, notTaken)

    return memo[idx][curr]


# -----------------------------------------------------------------------
"""
Delayed Projects

There are several projects, and each is denoted by a one letter name. Each project may depend on one or more other projects (or none). For example, if project A depends on project B, then project A cannot complete before project B. Suppose you are given a list L, of K such dependencies, and also a list D, of J projects that have been delayed. Output a list of all projects that will be delayed, in lexicographical (alphabetical) order. You can assume that a project, A, will be delayed if any project A depends on is delayed. The input is guaranteed to contain no circular dependencies.

Input:

Test cases will be provided in the following multiline format. The first line contains one integer, C, which is the number of test cases that will follow. Each test case has the following format.

The first line of a test case contains two integers, K and J, separated by a space. K is the number of dependencies, and J is the number of delayed projects. K lines follow, each with the format:

XY

where X and Y are the names of projects and project X depends on project Y, project names are single uppercase English 
letters. Each pair gives a project dependency: Y must complete before X can complete. All K lines together form the 
list L of project dependencies.

Finally, the last line contains J space-delimited project names (single letters, uppercase). This gives the list D of 
length J of projects that have been delayed. Each project in D is listed in the dependency list at least once.

Limits:

Test case count: 1 <= C <= 20
Number of dependencies: 1 <= K <= 100
Number of projects: 1 <= J <= 26
Project name: Each name is a single uppercase letter from A to Z.
Outputs:
For each test case, output one line containing the test case index, starting from 1, followed by a space-delimited 
list of projects that will be delayed, do not add any space at the end of each line of output. The list must be in 
lexicographically sorted order. The resulting line should be in this format:

Case #i: X[1] X[2]...

where i is the index of the test case, starting from 1, and X[k] are the names of the projects that were delayed.
"""


def getDelayedProjects(dependencies, delayedProjects):
    dependenciesLists = collections.defaultdict(list)

    for pair in dependencies:
        dependenciesLists[pair[1]].append(pair[0])  # project : [list of projects that depends on it]

    delayed = set()
    while len(delayedProjects) > 0:
        curr = delayedProjects.pop()
        delayedProjects.extend(dependenciesLists[curr])
        delayed.add(curr)
        for val in dependenciesLists[curr]:
            delayed.add(val)

    return delayed

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
A company needs to hire employees and to hire an employees, the company has to give a fixed fee to the employment agency,
also a fixed amount of severance pay is given to the employee when they are terminated - in addition to the monthly salary
they receive. You are given the following values hiring cost, salary of the employee, severance fee, number of months n,
and an integer array which contains minimum employees required by the company for each month for n number of months. 
Calculate and print the minimum cost to company which is required to keep minimum employees each month for n months.

Note: there are no employees on hand before the first month.

input:
5 lines:
- first 4 lines contain an integer which represents hiring cost, salary, severance fee and number of months respectively.
- fifth line contains array of n integers which represents minimum number of employees required by the company each month
for n months.

"""


def solver(cost, salary, severance, nums):
    dp = {0: 0}
    for req in nums:
        tmp = collections.defaultdict(lambda: float('inf'))
        for key in dp:
            if key >= req:
                for i in range(req, key + 1):
                    tmp[i] = min(tmp[i], dp[key] + i * salary + (key - i) * severance)
            else:
                tmp[req] = min(tmp[req], dp[key] + req * salary + (req - key) * cost)
        dp = tmp
    return min(dp.values())


# -----------------------------------------------------------------------
"""
Statistics
"""


def getDetails(fruits):
    details = {}

    for pair in fruits:
        fruit, price = pair[0], pair[1]
        if fruit not in details:
            details[fruit] = [price, price, 1, price]
        else:
            details[fruit][0] = min(details[fruit][0], price)
            details[fruit][1] = max(details[fruit][1], price)
            details[fruit][2] += 1
            details[fruit][3] += price

    for fruit in sorted(details):
        print(fruit, details[fruit][0], details[fruit][1], details[fruit][3] // details[fruit][2])

    # time O(n * log(n))
    # space O(n)


# -----------------------------------------------------------------------
"""
A pizza shop offers n pizzas along with m toppings. A customer plans to spend around x coins. The customer should order 
exactly one pizza, and may order zero, one or two toppings. Each topping may be ordered only once.

Given the lists of prices of available pizzas and toppings, what is the price closest to x of possible orders? Here, 
a price said closer to x when the difference from x is the smaller. Note the customer is allowed to make an order that 
costs more than x.

Example 1:

Input: pizzas = [800, 850, 900], toppings = [100, 150], x = 1000
Output: 1000
Explanation:
The customer can spend exactly 1000 coins (two possible orders).

Example 2:

Input: pizzas = [850, 900], toppings = [200, 250], x = 1000
Output: 1050
Explanation:
The customer may make an order more expensive than 1000 coins.

Example 3:

Input: pizzas = [1100, 900], toppings = [200], x = 1000
Output: 900
Explanation:
The customer should prefer 900 (lower) over 1100 (higher).

Example 4:

Input: pizzas = [800, 800, 800, 800], toppings = [100], x = 1000
Output: 900
Explanation:
The customer may not order 2 same toppings to make it 1000. 
Constraints:

Customer's budget: 1 <= x <= 10000
Number of pizzas: 1 <= n <= 10
Number of toppings: 0 <= m <= 10
Price of each pizza: 1 <= pizzas[i] <= 10000
Price of each topping: 1 <= toppings[i] <= 10000
The total price of all toppings does not exceed 10000.
"""


def minDiff(pizzas, toppings, x):
    comb = getCombinations(toppings)
    comb.sort()
    res = float('inf')
    for price in pizzas:
        res = getClosestPrice(price, res, comb, x)
        if res == x:
            return res
    return res

    # time O(m^2 + n * log(m)) -> m length of toppings, n length of pizzas
    # space O(m^2)


def getCombinations(toppings):
    res = []

    for i in range(len(toppings)):
        res.append(toppings[i])
        for j in range(i + 1, len(toppings)):
            res.append(toppings[i] + toppings[j])
    return res


def getClosestPrice(price, res, toppings, x):
    start, end = 0, len(toppings) - 1
    res = getMinDiff(x, price, res)
    while start <= end:
        mid = (start + end) // 2
        currPrice = price + toppings[mid]
        res = getMinDiff(x, res, currPrice)
        if currPrice < x:
            start = mid + 1
        elif currPrice > x:
            end = mid - 1
        else:
            return res

    return res


def getMinDiff(x, price1, price2):
    if abs(x - price1) < abs(x - price2):
        return price1
    elif abs(x - price1) == abs(x - price2):
        return price1 if price1 < price2 else price2

    return price2


# -----------------------------------------------------------------------
"""
Min Distance To The Farthest Node

You are given a tree-shaped undirected graph consisting of n nodes labeled 1...n and n-1 edges. 
The i-th edge connects nodes edges[i][0] and edges[i][1] together.
For a node x in the tree, let d(x) be the distance (the number of edges) from x to its farthest node. 
Find the min value of d(x) for the given tree.
The tree has the following properties:

It is connected.
It has no cycles.
For any pair of distinct nodes x and y in the tree, there's exactly 1 path connecting x and y.

Example 1:
Input: n = 6, edges = [[1, 4], [2, 3], [3, 4], [4, 5], [5, 6]]
Output: 2

Example 2:
Input: n = 6, edges = [[1, 3], [4, 5], [5, 6], [3, 2], [3, 4]]
Output: 2

Example 3:
Input: n = 10, edges = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
Output: 5

"""


def minDistance(edges, n):
    graph = collections.defaultdict(list)
    vertices = set()

    # initializing
    for edge in edges:
        u = edge[0]
        v = edge[1]
        vertices.add(u)
        vertices.add(v)
        graph[u].append(v)
        graph[v].append(u)

    minDist = float('inf')
    for v in vertices:
        minDist = min(minDist, getFarthestDistance(graph, vertices, v))
    return minDist

    # time O(n * (n+m)) -> n -> num of nodes, m -> num of edges
    # space O(n^2)


def getFarthestDistance(graph, vertices, v):
    status, distance, q = {}, {}, []

    for ver in vertices:
        status[ver] = 'w'
        distance[ver] = float('inf')

    q = [v]
    status[v] = 'g'
    distance[v] = 0
    maxDist = float('-inf')

    while len(q) > 0:
        curr = q.pop(0)
        adjacent = graph[curr]

        for ver in adjacent:
            if status[ver] == 'w':
                q.append(ver)
                status[ver] = 'g'
                distance[ver] = distance[curr] + 1
                maxDist = max(maxDist, distance[ver])

        status[curr] = 'b'

    return maxDist


# -----------------------------------------------------------------------
"""
cut tha cake
"""


def canWeCut(matrix, i, pattern):
    if pattern == 'hor':
        upperPiece = matrix[:i]
        lowerPiece = matrix[i:]

        if any([any(lst) for lst in upperPiece]) and any([any(lst) for lst in lowerPiece]):
            return True
        else:
            return False

    elif pattern == 'ver':
        # leftPiece
        isLeftHaveStraw, isRightHaveStraw = False, False
        for i_index in range(len(matrix)):
            for j_index in range(i):
                if matrix[i_index][j_index]:
                    isLeftHaveStraw = True
                    break

            if isLeftHaveStraw:
                break

        for i_index in range(len(matrix)):
            for j_index in range(i, len(matrix[0])):
                if matrix[i_index][j_index]:
                    isRightHaveStraw = True
                    break

            if isRightHaveStraw:
                break

        if isLeftHaveStraw and isRightHaveStraw:
            return True
        else:
            return False
    else:
        pass


def computeCutCombination(matrix, num_cuts):
    if num_cuts == 0: return 1

    num_combinations = 0
    for i in range(1, len(matrix)):
        if canWeCut(matrix, i, 'hor'):
            num_combinations += computeCutCombination(matrix[i:], num_cuts - 1)

    for j in range(1, len(matrix[0])):
        if canWeCut(matrix, j, 'ver'):
            matrix_ = [row[j:] for row in matrix]
            num_combinations += computeCutCombination(matrix_, num_cuts - 1)

    return num_combinations


def Solution(K, matrix):
    num_cuts = K - 1
    num_combinations = computeCutCombination(matrix, num_cuts)
    return num_combinations


# -----------------------------------------------------------------------
"""
best fruit

"""


def Solution(N, M, preferences):
    candidates = [i for i in range(1, N + 1)]
    for _ in range(N - 1):
        voting_count = {i: 0 for i in candidates}
        for preference in preferences:
            voting_count[preference[0]] += 1

        for candidate, _ in sorted(voting_count.items(), key=lambda x: (x[1], x[0])):
            candidates.remove(candidate)
            for preference in preferences:
                preference.remove(candidate)
            break

    return candidates[0]


# -----------------------------------------------------------------------
"""
Several coupons are placed in a row, and to win the prize you need to pick up at least two coupons of the same value.
Notice you can only pick up consecutive coupons from the row.

Given an array of integers representing the value of coupons, return the minimum number of consecutive coupons to pick up
to win the prize. Return -1 if its not possible.

Example:

input: coupons = [1,3,4,2,3,4,5,8]
output: 4
"""


def minPickUpCoupons(coupons):
    indexes = {}
    n = len(coupons)
    left, right = 0, 0
    res = float('inf')
    while right < n:
        if coupons[right] not in indexes:
            indexes[coupons[right]] = right
        else:
            res = min(res, right - indexes[coupons[right]] + 1)
            while left < n and coupons[right] in indexes:
                del indexes[coupons[left]]
                left += 1
            indexes[coupons[right]] = right

        right += 1

    return -1 if res == float('inf') else res

    # time O(n)
    # space O(n)


# -----------------------------------------------------------------------
"""
you are a gardener and you take care of youre plants

"""


def minNumOfSteps(plants, capacity):
    can = capacity

    p = 0
    counter = 1
    while p < len(plants):

        can -= plants[p]
        if p < len(plants) - 1 and can < plants[p + 1]:
            can = capacity
            counter = counter + ((p + 1) * 2)
        counter += 1
        p += 1

    return counter - 1

    # time O(n)
    # space O(1)


# -----------------------------------------------------------------------
"""
in an array of integers, array[i] denotes the error rate of server i.
an alert gets triggered for any one of the subarrays of size X, if all error rates in the subarray appear greater than
the ceil(threshold/X)

given array of integers denoting errors rate and integer for threshold implement a function to decide if there is a 
value X which triggers the alert.
"""

import math


def triggerAlert(errors, threshold):
    n = len(errors)
    for i in range(n):
        currMin = errors[i]
        for j in range(i, n):
            currMin = min(currMin, errors[j])
            size = j - i + 1
            if currMin > math.ceil(threshold / size):
                return True
    return False

    # time O(n^2)
    # space O(1)


# -----------------------------------------------------------------------
"""
Maximum Area Serving Cake

Given an array containing the radii of circular cakes and the number of guests, determine the largest piece that can be cut from the cakes such that every guest gets a piece of the cake with the same area. It is not possible that a single piece has some part of one cake and some part of another cake and each guest is served only one piece of cake.

Example 1
Radii = [ 1, 1, 1, 2, 2, 3] numberOfGuests = 6.
Output: 7.0686

Reason being you can take the area of the cake with a radius of 3, and divide by 4. (Area 28.743 / 4 = 7.0686)
Use a similary sized piece from the remaining cakes of radius 2 because total area of cakes with radius 2 are > 7.0686

Example 2
Radii [4, 3, 3] numberOfGuests = 3
Output: 28.2743

Example 3
Radi [6, 7] numberOfGuests = 12
Output: 21.9911
"""


def maxAreaServing(radiuses, guests):
    areas = [math.pi * pow(r, 2) for r in radiuses]

    left, right = 0, max(areas)
    while round(left, 4) < round(right, 4):
        mid = (left + right) / 2

        if canServe(mid, areas, guests):
            left = mid
        else:
            right = mid

    return round(left, 4)

    # time O(n * log(max_radius)) -> n num of radiuses
    # space O(n)


def canServe(currArea, areas, guests):
    counter = 0
    for area in areas:
        counter += area // currArea
    return counter >= guests
