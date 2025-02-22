---
layout:     post
title:      Daily Leetcode Post
date:       2020-04-09 12:32:18
summary:    Just Practicing DSA Questions
categories: jekyll pixyll
---

## All elements in two binary search trees.

Given two binary search trees `root1` and `root2`, return *a list containing all the integers from both trees sorted in **ascending** order*.

##### Solution 1.

Intuitively this problem is very easy to solve and pretty straightforward if we are familiar with any of the depth first search traversals of a binary tree. 

The idea behind the solution is simple --> 

1. do a dfs traversal on tree1 and store values in list1.
2. do a dfs traversal on tree2 and store values in list2.
3. merge the two lists and return.

```python
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        
        def dfs(root, array):
            if root:
                dfs(root.left, array)
                array.append(root.val)
                dfs(root.right, array)
            return array
        
        a1, a2 = dfs(root1, []), dfs(root2,[])
        return sorted(a1 + a2)
```

Additionally in Python the dfs traversal can be made as a cool one liner, as -

```python
def dfs(root, array):
    return dfs(root.left, array) + [root.val] + dfs(root.right, array) if root else []
```

The time complexity of this algorithm is O(N+M)log(N+M) where N is the number of nodes in the first tree and M is the number of nodes in the second tree. Space complexity is O(N+M) to keep the output and O(h)-> space for implicit stack (h = height of the tree).

This solution can be further improved from linearithmic time to linear.

The idea is to do iterative inorder depth first search. 

![Source Leetcode](https://leetcode.com/problems/all-elements-in-two-binary-search-trees/Figures/1305/iterative.png)

The algorithm is as follows:

1. Initialize two stacks and an output list.
2. While either of the roots are still not null or either of the stacks are still not empty - do -->
3. Move to the leftmost node of each root1 and root2. This will give the minimum value in inorder traversal. While going to the leftmost value, push the root node into the stack. 
4. As soon as we reach the leftmost nodes for that particular root, compare the tops of both stacks and according append in the output list. If at all one node is popped from the stack then make the root equal to its right node and continue the inorder traversal.
5. Return the list.

```python
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        
        stack1, stack2, ans = [], [], []
        
        while root1 or root2 or stack1 or stack2:
            while root1:
                stack1.append(root1)
                root1 = root1.left
            while root2:
                stack2.append(root2)
                root2 = root2.left
            if not stack2 or stack1 and stack1[-1].val <= stack2[-1].val:
                root1 = stack1.pop()
                ans.append(root1.val)
                root1 = root1.right
            else:
                root2 = stack2.pop()
                ans.append(root2.val)
                root2 = root2.right
        return ans
```

The time complexity for this problem is O(M+N) and space complexity is O(M+N).

## Number of Provinces.

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        
        n = len(isConnected)
        graph = defaultdict(list)
        for i in range(n):
            for j in range(n):
                if isConnected[i][j]:
                    graph[i].append(j)
        seen = set()
        provinces = 0
        def dfs(i):
            edges = graph[i]
            seen.add(i)
            for node in edges:
                if node not in seen:
                    dfs(node)
        for i in range(n):
            if i not in seen:
                dfs(i)
                provinces+=1
        return provinces
```



```c++
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) 
    {
        unordered_set <int> st;
        queue <int> q;
        int n = isConnected.size();
        int count = 0;
        for(int i = 0; i < n; i++)
        {
            if(st.find(i) == st.end())
            {
                q.push(i);
                while(!q.empty())
                {
                    int node = q.front();
                    q.pop();
                    st.insert(node);
                    for(int j = 0; j < n; j++)
                    {
                        if(isConnected[node][j] == 1 && st.find(j) == st.end())
                        {
                            q.push(j);
                        }
                    }
                }
                count++;       
            }
        }
        return count;
    }
};
```

## House Robber - Dynamic Programming.

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

The greedy solution to the problem is as follows ->

```c++
class Solution {
public:
    int rob(vector<int>& nums) 
    {
        int prev = 0;
        int loot = nums[0];
        for(int i = 1; i < nums.size(); i++)
        {
            int temp = loot;
            loot = max(nums[i] + prev, loot);
            prev = temp;
        }
        return loot;
    }
};
```



```python

```

## Longest Common Subsequence



```python

```

## Unique Paths

There is a robot on an `m x n` grid. The robot is initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.

Given the two integers `m` and `n`, return *the number of possible unique paths that the robot can take to reach the bottom-right corner*.

The test cases are generated so that the answer will be less than or equal to `2 * 109`.

This is a standard dynamic programming task. 

### Algorithm :



![](https://leetcode.com/problems/unique-paths/Figures/62/bin4.png)

**Algorithm**

- Initiate 2D array `d[m][n] = number of paths`. To start, put number of paths equal to 1 for the first row and the first column. For the simplicity, one could initiate the whole 2D array by ones.
- Iterate over all "inner" cells: `d[col][row] = d[col - 1][row] + d[col][row - 1]`.
- Return `d[m - 1][n - 1]`.

```c++
class Solution {
public:
    int uniquePaths(int m, int n) 
    {
        vector<vector<int>> dp(m, vector<int> (n, 1));
        int count = 0;
        for(int i = 1; i < m; i++)
        {
            for(int j = 1; j < n; j++)
            {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```



```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)]
        for col in range(1, m):
            for row in range(1, n):
                dp[col][row] = dp[col-1][row] + dp[col][row-1]
        return dp[m-1][n-1]
        
```



## Unique Paths II

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        
        rows, cols = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for j in range(cols):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        for i in range(1, rows):
            for j in range(1, cols):
                if obstacleGrid[i][j]: continue
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```

```c++
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) 
    {
        int rows = obstacleGrid.size(); int cols = obstacleGrid[0].size();
        vector <vector <int>> dp(rows, vector<int>(cols, 0));
        for(int r = 0; r < rows; r++)
        {
            if ( obstacleGrid[r][0] == 1 )
                break;
            dp[r][0] = 1;
        }
        for(int c = 0; c < cols; c++)
        {
            if ( obstacleGrid[0][c] == 1 )
                break;
            dp[0][c] = 1;
        }
        for(int r = 1; r < rows; r++)
        {
            for(int c = 1; c < cols; c++)
            {
                if(obstacleGrid[r][c] == 1)
                    continue;
                else
                    dp[r][c] = dp[r-1][c] + dp[r][c-1];
            }
        }
        return dp[rows-1][cols-1];
    }
};
```

