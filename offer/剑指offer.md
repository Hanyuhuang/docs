# 1、二维数组中的查找

## 题目描述

> 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

## 题解

**方法一：** 遍历整个二维数组，如果遍历完还没有找到该数就没有，代码略。

**方法二：**从二维数组的右上角或左下角开始遍历。以从右上角遍历为例：数组中的数为`array[i][j]`，要找的数为`target`。如果`array[i][j]` > `n`，根据数组特征，该位置下面的数都比`array[i][j]`大，左边的数都比`array[i][j]`小，所以继续往左遍历。如果`array[i][j]` <` n`，则继续往下遍历，依次类推，代码如下。

```java
public class Solution {
    public boolean Find(int target, int[][] array) {
        if (array == null || array.length < 1 || array[0].length < 1) {
            return false;
        }
        int row = array.length;
        int col = array[0].length;
        // 从右上角开始寻找（左下角开始也可以）
        for (int i = 0; i < row; i++) {
            for (int j = col - 1; j >= 0; j--) {
                if (array[i][j] == target) {
                    return true;
                } else if (array[i][j] < target) {
                    break; // 下一行
                }
            }
        }
        return false;
    }
}
```

# 2、替换空格

## 题目描述

> 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

## 题解

**方法一：**使用现成的Api，String.replace()

```java
public class Solution {
    public String replaceSpace(StringBuffer str) {
    	  // 直接调用api
        return str.toString().replace(" ","%20");
    }
}
```

**方法二：**遍历一次，把空格替换成对应的字符串即可。

```java
public class Solution {
    public String replaceSpace(StringBuffer str) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(str.charAt(i));
            }
        }
        return sb.toString();
    }
}
```

# 3、从尾到头打印链表

## 题目描述

> 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。

## 题解

**方法一：非递归**

```java
import java.util.*;
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> list = new ArrayList();
        while(listNode != null){
            list.add(listNode.val);
            listNode = listNode.next;
        }
        Collections.reverse(list);
        return list;   
    }
}
```

**方法二：递归**

```java
import java.util.*;
public class Solution {
  
    private ArrayList<Integer> list = new ArrayList();
  
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode != null){
        		printListFromTailToHead(listNode.next);
          	list.add(listNode.val);
        }
        return list;   
    }
}
```

# 4、重建二叉树

## 题目描述

> 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

## 题解

前序遍历数组中的第一个数为二叉树的根节点，该根节点在中序遍历数组将数组分成两部分，左边的部分为该根节点的左孩子节点，右边的部分为该根节点的右孩子节点，将左边和右边再分别依次类推。

```java
/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        return rebuild(pre, in, 0, pre.length - 1, 0, in.length - 1);
    }

    public TreeNode rebuild(int[] pre, int[] in, int preStart, int preEnd, int inStart, int inEnd) {
        if (preStart > preEnd || inStart > inEnd) return null;
        // 构建新节点
        TreeNode root = new TreeNode(pre[preStart]);
        // 找到根节点在中序遍历数组中的位置
        int index = 0;  
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == pre[preStart]) { 
                index = i;
                break;
            }
        }
        int offSet = index - inStart; //偏移量
        root.left = rebuild(pre, in, preStart + 1, preStart + offSet, inStart, index - 1);
        root.right = rebuild(pre, in, preStart + offSet + 1, preEnd, index + 1, inEnd);
        return root;
    }
}
```

# 5、用两个栈实现队列

## 题目描述

> 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

## 题解

两个栈1和2，当将入元素时都放入栈1中，当需要弹出元素都从栈2中弹出。如果要弹出元素时，栈2中没有元素，就将栈1中的所有元素都弹出到栈2中，然后再从栈2顶部弹出元素。

```java
import java.util.Stack;
public class Solution {
    private Stack<Integer> stack1 = new Stack<Integer>();  //存放新加入的数
    private Stack<Integer> stack2 = new Stack<Integer>();  //存放待弹出的数

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (stack2.isEmpty()) {
          	// 将stack1中的数全部弹出 放入到stack2中
            while (!stack1.isEmpty()) { 
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }
}
```

# 6、旋转数组的最小数字

## 题目描述

> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
> 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
> 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
> NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

## 题解

**方法一：**优化后的暴力法。根据旋转后数组的特征，两段非递减的数组，分界处较小的数就是原始数组中最小的数。

```java
public class Solution {
    public int minNumberInRotateArray(int[] array) {
        for (int i = 0; i < array.length - 1; i++) {
            if (array[i] > array[i + 1]) {
                return array[i + 1];
            }
        }
        return 0;
    }
}
```

**方法二：**二分思想，两段开始位置的数应该大于结束位置的数，如果不符合这个条件就继续二分，直到找到对应的数。

```java
public class Solution {
  	public int minNumberInRotateArray(int[] array) {
        int left = 0, right = array.length - 1;
        // 这里不能加上等于的情况，因为剩下一个数的时候无法判断
        while (left < right) {
            // 二分结束的点是最后剩下一个数组长度为2的时候
            if (left + 1 == right && array[left] > array[right]) {
                return array[right];
            }
            int mid = (right + left) / 2;
            if (array[mid] >= array[left]) {
                // 不能用mid+1，可能mid正好为分界处，最后需要比较
                left = mid;
            } else {
                right = mid;
            }
        }
        return 0;
    }
}
```

# 7、斐波那契数列

## 题目描述

> 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
>
> n<=39

## 题解

**方法一：**递归

```java
public class Solution {
    public int Fibonacci(int n) {
        if (n <= 1) return n;
        return Fibonacci(n-1) + Fibonacci(n-2);
    }
}
```

**方法二：**非递归(动态规划)

```java
public class Solution {
		public int Fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[n] = dp[n - 1] + dp[n - 2];
        }
        return dp[n];
    }
}
```

**方法三：**优化后的动态规划

```java
public class Solution {
    public int Fibonacci(int n) {
        int pre = 0, next = 1, res = n;
        for (int i = 2; i <= n; i++) {
            res = pre + next;
            pre = next;
            next = res;
        }
        return res;
    }
}
```

**方法四：**使用两个变量的动态规划

```java
public class Solution {
    public int Fibonacci(int n) {
      	if (n <= 1) {
            return n;
        }
        int pre = 0, res = 1;
        for (int i = 2; i <= n; i++) {
            res += pre;
          	pre = res - pre;
        }
        return res;
    }
}
```

# 8、跳台阶

## 题目描述

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

## 题解

还是斐波那契数列，只是这里f(2)的值不一样，直接给出最优解。

```java
public class Solution {
    public int JumpFloor(int target) {
      if (target <= 2) {
        	return target;
      }
      int pre = 1, res = 1;
      for (int i = 2; i <= target; i++) {
       		 res += pre;
       		 pre = res - pre;
      }
      return res;
    }
}
```

# 9、变态跳台阶

## 题目描述

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

## 题解

+ f(n) = f(n-1) + f(n-2) + … + f(2) + f(1)
+ f(n-1) = f(n-2) + … + f(2) + f(1)

两式相减得：f(n) = 2f(n-1)

```java
public class Solution {
    public int JumpFloorII(int n) {
        return 1 << (n-1);
    }
}
```

# 10、矩形覆盖

## 题目描述

> 我们可以用2\*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2\*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

## 题解

找规律，最后发现还是一个斐波那契数列。

```java
public class Solution {
    public int RectCover(int n) {
        if (n <= 2) {
            return n;
        }
        int pre = 1, res = 1;
        for (int i = 2; i <= n; i++) {
            res += pre;
            pre = res - pre;
        }
        return res;
    }
}
```

# 11、二进制中1的个数

## 题目描述

> 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

## 题解

与：`1|1=1`,`1|0=1`,`0|0=0`

或：`1&1=1`,`1&0=0`,`0&0=1`

算术右移：可以理解为二进制去掉最后一位，第一位前面加个0

```java
public class Solution {
    public int NumberOf1(int n) {
        int res = 0; // 统计1的个数
        while(n != 0){
            res += n&1; //最低位是否为1
            n >>>= 1;   // 整体算数右移一位
        }
        return res;
    }
}
```

# 12、数值的整数次方

## 题目描述

> 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
>
> 保证base和exponent不同时为0

## 题解

题目意思就是求 $x^{n}$，最简单思路暴力法从1遍历到n，每次都乘一次x。下面使用优化的方法。

当n为偶数时：$x^{n}=(x^{\frac{n}{2}})^2​$

当n为奇数时：$x^{n}=(x^{\frac{n}{2}})^2*x​$

```java
public class Solution {
    public double Power(double base, int exponent) {
        if (exponent < 0) {
            base = 1 / base;
            exponent = -exponent;
        }
        return pow(base, exponent);
    }

    private double pow(double x, int n) {
        if (n == 0) {
          	return 1;
        }
        double half = pow(x, n / 2);
        if (n % 2 == 1) {
          	return half * half * x;
        } else {
         	return half * half;
        }
    }
}
```

# 13、调整数组顺序使奇数位于偶数前面

## 题目描述

> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

## 题解

遍历数组，使用两个变量`i`和`j`，`i`为偶数所在的位置，`j`为下一个奇数所在的位置，当`j`找到下一个奇数时，交换`i`和`j`位置上的两个数，然后将原来在`i`位置上的偶数插入到交换后的奇数后面，剩余的偶数依次往后移动。如果遍历完整个数组都没有找到下一个奇数，说明后面的数都为偶数，遍历结束。

```java
public class Solution {
    public void reOrderArray(int[] array) {
        for (int i = 0, j = 0; i < array.length; i++) {
            if (array[i] % 2 == 1) {
                continue;
            }
            // 找到下一个奇数
            j = i + 1;
            while (j < array.length && array[j] % 2 != 1) {
                j++;
            }
            if (j >= array.length) {
                break;
            }
            // 交换数组位置
            int temp = array[j];
            for (int k = j; k > i; k--) {
                array[k] = array[k - 1];
            }
            array[i] = temp;
        }
    }
}
```

# 14、链表中倒数第K个结点

## 题目描述

> 输入一个链表，输出该链表中倒数第k个结点。

## 题解

快慢指针，快指针比慢指针先走k步，然后同时走，当快指针遍历完的时候，慢指针所在的位置就是倒数第k个结点。

```java
public class Solution {
    public ListNode FindKthToTail(ListNode head, int k) {
        ListNode fast = head, slow = head;
        while (k-- > 0) {
            if (fast == null){
                return null;
            }
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

# 15、反转链表

##  题目描述

> 输入一个链表，反转链表后，输出新链表的表头。

## 题解

**非递归**

```java
public class Solution {
    public ListNode ReverseList(ListNode head) {
        ListNode pre = null;  // 用来保存上一个节点
        ListNode temp = null; // 保存下一个要遍历的节点
        while (head != null){
            temp = head.next; // 保存下一个要遍历的节点
            head.next = pre;  // 当前节点指向前一个节点
            pre = head;       // 更新前一个节点为当前节点
            head = temp;      // 更新当前节点为下一个要遍历的节点
        }
        return pre;
    }
}
```

**递归**

```java
public class Solution {
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null){
          	return head;
        }
        // 下一个节点
        ListNode temp = ReverseList(head.next);
      	head.next.next = head;
      	head.next = null;
      	return temp;
    }
}
```

# 16、合并两个排序的链表

## 题目描述

> 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

## 题解

使用归并排序的思想

```java
public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {
        ListNode list = new ListNode(0);
        ListNode temp = list;
        while (list1 != null && list2 != null){
            // temp的后面需要接上list1，list2中较小的
            temp.next = list1.val <= list2.val ? list1 : list2;
            // temp的后面接的是list1还是list2 需要遍历下一个
            if(temp.next == list1){
              	list1 = list1.next;
            } 
            else {
             		list2 = list2.next;
            }
            temp = temp.next;
        }
        // list1 和 list2 中必有一个未遍历完  接到temp后面
        if (list1 == null){
          	temp.next = list2;
        } else{
          	temp.next = list1;
        } 
        return list.next;
    }
}
```

# 17、树的子结构

## 题目描述

> 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

## 题解

遍历树A，当A的节点等于B的根节点时，判断以当前A节点为根节点的树与树B是否重合。

```java
public class Solution {
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        if (root1==null || root2==null) return false;
        //  当前节点root1是否匹配 || 递归匹配root1的左右孩子节点  
        return isSubtree(root1,root2) || HasSubtree(root1.left,root2) || HasSubtree(root1.right,root2);
    }
    
    private boolean isSubtree(TreeNode root1,TreeNode root2){
        if (root2 == null) return true; // 匹配结束 
        if (root1 == null || root1.val != root2.val) return false; 
        // 当前值相等  开始递归匹配左右孩子
        return isSubtree(root1.left,root2.left) && isSubtree(root1.right,root2.right);
    }
}
```

# 18、二叉树的镜像

## 题目描述

> 操作给定的二叉树，将其变换为源二叉树的镜像。

## 输入描述:

```
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```

## 题解

使用一个临时遍历保存左节点或右节点然后交换。

```java
public class Solution {
    public void Mirror(TreeNode root) {
        if (root == null) return;
        // 交换左右孩子节点
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        // 递归继续交换
        Mirror(root.left);
        Mirror(root.right);
    }
}
```

# 19、顺时针打印矩阵

## 题目描述

> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

## 题解

此题没有什么快捷的方法，只能老老实实遍历，但需要理清思路遍历。我们这里遍历的思路是每行或每列遍历的时候，留下最后一个，别遍历完。

```java
import java.util.ArrayList;

public class Solution {
  
    public ArrayList<Integer> printMatrix(int [][] matrix) {
        ArrayList<Integer> list = new ArrayList();
        if (matrix == null || matrix.length < 1 || matrix[0].length < 1) return list;
        int row = matrix.length;
        int col = matrix[0].length;
        int i = 0, j = 0;
        while (i < row && j < col){
            print(list, matrix, i++, row--, j++, col--);
        }
        return list;
    }
    // 每一圈具体打印方法
    public void print(ArrayList<Integer> list,int[][] matrix,int rowStart,int rowEnd,int colStart,int colEnd){
        if (rowStart == rowEnd){  // 只剩一行
            for (int i = colStart;i <= colEnd;i++) list.add(matrix[rowStart][i]);
        } else if (colStart == colEnd){  // 只剩一列
            for(int i = rowStart;i <= rowEnd;i++) list.add(matrix[i][colStart]);
        } else{
            // 打印最上面一行 除去最右边的数
            for(int i = colStart;i < colEnd;i++) list.add(matrix[rowStart][i]);
            // 打印最右边一列 除去最下面的数
            for(int i = rowStart;i < rowEnd;i++) list.add(matrix[i][colEnd]);
            // 逆序打印最下边一行 除去最左边的数
            for(int i = colEnd;i > colStart;i--) list.add(matrix[rowEnd][i]);
            // 逆序打印最左一列 除去最上面的数
            for(int i = rowEnd;i > rowStart;i--) list.add(matrix[i][colStart]);
        }
    }
}
```

# 20、包含min函数的栈

## 题目描述

> 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

## 题解

使用两个栈，一个栈保存正常数据，另一个为单调栈，栈中元素顺序为非递增。当栈中加入元素时，判断新元素加入到单调栈中是否满足非递增的条件，如果满足就加入，否则就放弃。同样的，当弹出元素时，要判断弹出的元素是否为单调栈栈顶的元素，如果是，则需要一起弹出。

```java
import java.util.Stack;

public class Solution {
    
    Stack<Integer> stack = new Stack();  
    Stack<Integer> minStack = new Stack();
    
    public void push(int node) {
        stack.push(node);
        // 保证最小栈从大到小
        if(minStack.isEmpty() || node <= minStack.peek()){
          	minStack.push(node);
        } 
    }
    
    public void pop() {
        // 最小栈的栈顶对应的元素被弹出 最小栈中也需要弹出
        if(minStack.peek() == stack.pop()){
            minStack.pop();
        } 
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int min() {
        return minStack.peek();
    }
}
```

# 21、栈的压入、弹出序列

## 题目描述

> 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

## 题解

使用一个栈，将入栈序列都放入到栈中，一旦栈顶元素与出栈序列当前数相等就匹配出栈，遍历结束后判断栈中是否还有元素就能确定是否是出栈序列。

```java
import java.util.Stack;

public class Solution {
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        Stack<Integer> stack = new Stack();
        for(int i = 0,j = 0;i < pushA.length;i++){
            stack.push(pushA[i]);
            while(!stack.isEmpty() && popA[j] == stack.peek()){
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }
}
```

# 22、从上往下打印二叉树

## 题目描述

> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。

## 题解

二叉树的层次遍历

```java
import java.util.*;

public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> list = new ArrayList();
        if (root == null) return list;
        LinkedList<TreeNode> queue = new LinkedList();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            for (int i = 0;i < size;i++){
                root = queue.poll();
                list.add(root.val);
                if (root.left != null){
                  	queue.offer(root.left);
                } 
                if (root.right != null){
                  	queue.offer(root.right);
                } 
            }
        }
        return list;
    }
}
```

# 23、二叉树的后序遍历序列

## 题目描述

> 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

## 题解

一个正常遍历二叉搜索树后序遍历结果的数组应该满足这样的规则：数组最后一位是该数的根节点，然后可以将数组剩余部分划分成两部分，左边部分的数都比根节点小，右边部分的数都比根节点大。根据这个特征，我们可以先找到左边的部分，然后验证剩余的部分是否都比根节点的数大。如果不满足可以直接确定，如果满足就继续递归判断。

```java
public class Solution {
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length < 1) {
            return false;
        }
        return isBST(sequence, 0, sequence.length - 1);
    }

    // 根节点 将数组分成两部分 左边都比他小 右边都比他大
    public boolean isBST(int[] a, int left, int right) {
        if (left >= right) {
            return true;
        }
        int i = left;
        while (a[i] < a[right]) {
            i++;   // 确定左边
        }
        for (int j = right - 1; j >= i; j--) {   //验证后边是否都比根节点大
            if (a[j] < a[right]) {
                return false;
            }
        }
        return isBST(a, left, i - 1) && isBST(a, i, right - 1);
    }
}
```

# 24、二叉树中和为某一值的路径

## 题目描述

> 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

## 题解

典型的回溯，先找到所有符合条件的路径，最后将list排序。

```java
import java.util.*;
public class Solution {
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        ArrayList<ArrayList<Integer>> list = new ArrayList();
        ArrayList<Integer> temp = new ArrayList();
        dfs(list,temp,root,target);
        // 排序 长度大的靠前
        Collections.sort(list,(o1,o2) -> o2.size() - o1.size());
        return list;
    }
    
    public void dfs(ArrayList<ArrayList<Integer>> list,ArrayList<Integer> temp,TreeNode root,int target){
        if (root == null) return;
        temp.add(root.val);
        target -= root.val;
        if(target == 0 && root.left == null && root.right==null) list.add(new ArrayList(temp));
        dfs(list,temp,root.left,target);    
        dfs(list,temp,root.right,target); 
        // 回溯 遍历完后把之前添加到temp中的数全部删除
        temp.remove(temp.size()-1);
    }
}
```

# 25、复杂链表的复制

## 题目描述

> 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

## 题解

**额外空间复杂度为O(n)的解法：**

使用一个Map保存每个节点和对应的特殊指针指向的节点。

```java
import java.util.*;
public class Solution {
  	public RandomListNode Clone(RandomListNode pHead){
      	if (pHead == null) return null;
        Map<RandomListNode,RandomListNode> map = new HashMap();
        RandomListNode cur = pHead,copy = null;
        while (cur != null){    // 复制结点
			copy = new RandomListNode(cur.label); // 创建一个新的结点
            map.put(cur,copy);   // 放入map中
            cur = cur.next;     // 继续遍历
        }
        for(RandomListNode node : map.keySet()){  // 遍历map
			map.get(node).next = map.get(node.next); // 复制节点的下一个指针
          	map.get(node).random = map.get(node.random);//复制节点的随机指针
        }
      	return map.get(pHead);
    }
}
```

**额外空间复杂度为O(1)的解法：**

遍历原始链表，在每个节点后面添加一个复制节点。在遍历一次，将每个原始节点的随机节点复制下来。最后再遍历一次，将所有的复制节点从原链表上分离出来。

```java
import java.util.*;
public class Solution {
  	public RandomListNode Clone(RandomListNode pHead){
      	if (pHead == null) return null;
        RandomListNode cur = pHead;
        // 在每个节点后面添加一个复制节点
        while (cur != null){
            RandomListNode copy = new RandomListNode(cur.label);
            copy.next = cur.next;
            cur.next = copy;
            cur = copy.next;
        }
        // 将每个复制节点的random指针指向其他复制节点
        cur = pHead;
        while (cur != null){
            cur.next.random = cur.random == null ? null : cur.random.next;
            cur = cur.next.next;
        }
        // 将复制的链表从原链表上分离下来
        RandomListNode head = pHead.next;
        cur = pHead;
        while (cur != null){
            RandomListNode copy = cur.next;
            cur.next = copy.next;
            cur = cur.next;
            copy.next = cur == null ? null : cur.next;
        }
        return head;
    }
}
```

# 26、二叉搜索树与双向链表

## 题目描述

> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

## 题解

使用中序遍历，到达该树最左边的叶子节点作为该链表的头结点，然后连接起来。

```java
import java.util.*;
public class Solution {
    public TreeNode Convert(TreeNode root) {
        if (root == null) return null;
        Stack<TreeNode> stack = new Stack();
        TreeNode head = null; // 链表的头结点
        TreeNode pre = null;  // 保存上一个结点
        while (root != null || !stack.isEmpty()){ // 中序遍历非递归
            if (root != null){
                stack.push(root);
                root = root.left; // 更新下个结点
            } else{
                root = stack.pop();
                if (head == null){   // 链表头结点 特殊处理
                    head = root;  // 记录头结点
                    pre = root;   // 保存上一个节点
                } else{    // 正常情况
                    root.left = pre;   // 当前节点的前一个节点为上一个节点
                    pre.right = root;  // 上一个节点的下一个节点为当前节点
                    pre = root;        // 保存上一个节点
                }          
                root = root.right; // 更新下一个节点
            }
        }
        return head;
    }
}
```

# 27、字符串的排序

## 题目描述

> 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

## 输入描述

> 输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。

## 题解

使用一个标记数组，记录每次遍历的时候某个位置的字母有没有被遍历过，然后使用回溯。先将字符串按照字母顺序排序，方便去除重复的情况。

```java
public class Solution {
        public ArrayList<String> Permutation(String str) {
            ArrayList<String> list = new ArrayList();
            if (str == null || str.trim().length() < 1) return list;
            char[] c = str.toCharArray();
            boolean[] flag = new boolean[c.length];
            Arrays.sort(c);
            dfs(list, c, "", 0, flag);
            return list;
        }

        public void dfs(ArrayList<String> list, char[] c, String res, int index, boolean[] flag) {
            if (index == c.length) {
                list.add(new String(res));
                return;
            }
            for (int i = 0; i < c.length; i++) {
                // 重要代码
                if (flag[i] || (i > 0 && c[i] == c[i - 1] && !flag[i - 1])) continue;
                flag[i] = true;
                res += c[i];
                dfs(list, c, res, index + 1, flag);
                flag[i] = false;
                res = res.substring(0, res.length() - 1);
            }
        }
    }
```

对上面有注释那行代码的if判断解释一下：

```java
if (flag[i] || (i > 0 && c[i] == c[i - 1] && !flag[i - 1])) continue;
```

+ flag[i] 很容易理解，i位置上的字母在此次以x为首字母的遍历中已经使用过。
+ i > 0 && c[i] == c[i - 1] && !flag[i - 1] 是防止重复的情况。比如字符串是aab，则会出现两个aab的情况，分别是123和213，第一次aab的时候，也就是123的遍历顺序， flag[i-1] 是true。第二次aab的时候，也就是213的顺序，flag[i-1] 就是 false。能确保这种重复的情况下，只有第一次遍历时才有效。

# 28、数组中出现次数超过一半的数字

## 题目描述

> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

## 题解

数组中，众数记为1，其他数记为-1，将这些数加起来，结果应该大于0。这个思路肯定能找到众数，但是数组中可能不存在众数，导致这样算最后也会找出来一个数，所以我们需要再验证一次，这个数是否在数组中出现了一半的次数。

```java
public class Solution {
    public int MoreThanHalfNum_Solution(int[] array) {
        int res = 0, count = 0;
        for (int i : array) {
            if (count == 0) {
                res = i;
            }
            count += res == i ? 1 : -1;
        }
        // 处理数组中没有数字超过半数的情况
        count = 0;
        for (int i : array) {
            if (i == res) {
                count++;
            }
        }
        if (count <= array.length / 2) {
            res = 0;
        }
        return res;
    }
}
```

# 29、最小的K个数

## 题目描述

> 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

## 题解

```java
import java.util.*;
public class Solution {
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> list = new ArrayList();
        if (input == null || input.length < 1 || input.length < k) return list;
        Arrays.sort(input);
        for (int i = 0; i < k; i++) {
            list.add(input[i]);
        }
        return list;
    }
}
```

# 30、连续子数组的最大和

## 题目描述

> HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

## 题解

动态规划，dp[i]为以i结尾的最大和，dp[i+1] = Math.max(dp[i],dp[i+1]+array[i]）

```java
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
        int res = 0;
        int[] dp = new int[array.length];
        dp[0] = array[0];
        for(int i = 1;i < array.length;i++){
            // 前i项和 与 第i项 比较
            dp[i] = Math.max(dp[i-1],dp[i-1]+array[i]);
            // 之前最大和 和 当前最大和比较
            res = Math.max(res,dp[i]);
        }
        return res;
    }
}
```

因为每次只和上一个数有关，可以优化只用一个变量记录上次i-1结尾的最大值。

```java
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
        int res = array[0];
        int max = array[0];
        for(int i = 1;i < array.length;i++){
            // 前i项和 与 第i项 比较
            max = Math.max(array[i],max+array[i]);
            // 之前最大和 和 当前最大和比较
            res = Math.max(res,max);
        }
        return res;
    }
}
```

# 31、整数中1出现的次数(从1到n整数中1出现的次数)

## 题目描述

> 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

## 题解

**思路：**分别计算个位、十位、百位。。。上1的个数，最后相加起来。

**个位上1的个数：** 这样的数为1、11、21、31…每隔10出现一次。比如n为22，出现22/10+1=3次，公式为n/10+1。要注意n的个位数，如果个位数为0，比如n为20，那么就是n/10。可以归纳为n/10+n%10。

**十位上1的个数：**这样的数为10、11、…19、110、111、…119…每隔100出现10次。比如n为222，出现了 (222/100)\*10+1\*10=30次。要注意n的十位数，如果十位数为0，比如n为202，那么就是(n/100)\*10。如果十位数大于等于2，就是刚计算的公式(n/100)*10+10。如果十位数为1，比如n为212，那么就是(n/100)\*10+(n%10+1)。

**百位上1的个数：**这样的数为100、101、…199、1100、1101、…、1199…每隔1000出现100次。比如n为2222，出现了(2222/1000)\*100+1\*100=300次。要注意百位上的数，如果百位数为0，那么就是(n/1000)\*100。如果百位数大于等于2，就是刚计算的公式(n/1000)*100+100。如果百位数为1，比如2122，那么就是(n/1000)\*100+(n%100+1)。

通过上面可以总结出一个通用的公式，如果用i表示个、十、百位，那么对应的值分别为1、10、100，最后一般公式为(n/i\*10)\*i+i。然后要考虑i位上数是0的情况，1的情况，其他情况。如果是0，那么公式为(n/i\*10)\*i。如果是1，那么公式为(n/i\*10)\*i+(n%i+1)。

```java
public class Solution {
    public int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        for (int i = 1; i <= n; i = i * 10) {
            int j = n % (i * 10);
            if (j < i) {
                count += (n / (i * 10)) * i;
            } else if (j >= 2 * i) {
                count += (n / (i * 10)) * i + i;
            } else {
                count += (n / (i * 10)) * i + (n % i + 1);
            }
        }
        return count;
    }
}
```

# 32、把数组排成最小的数

## 题目描述

> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

## 题解

将首位最小的数排在前面。比如在数组先找1开头的数，然后在1开头的数中再找后面一位是0的，依次类推，最后拼接起来。

```java
import java.util.*;
public class Solution {
    public String PrintMinNumber(int [] numbers) {
        String[] str = new String[numbers.length];
        for(int i = 0;i < numbers.length;i++){         // 将数字转成字符串
            str[i] = String.valueOf(numbers[i]);
        }
        //  自定义比较器  排好序
        Arrays.sort(str,(o1,o2)->Integer.parseInt(o1+o2)-Integer.parseInt(o2+o1));
        StringBuilder sb = new StringBuilder();
        for(String s : str){  // 拼接字符串
            sb.append(s);
        }
        return sb.toString();
    }
}
```

# 33、丑数

## 题目描述

> 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

## 题解

要保证丑数，就要用2,3,5这三个数分别去乘已有的丑数。一开始只有个1，所以要用2,3,5分别乘上1得到2,3,5，然后再分别用2,3,5乘上2。但是要从小到大排序，我们就要记住2,3,5乘的是哪个数。比如一开始2,3,5分别去乘1，结果不是1,2,3,5而是1,2,3,4,5。第一轮5还没有乘到1，第二轮2已经开始乘上2了，第三轮就应该是2,3,5去乘3，所以要记住这三个数分别到第几轮了。每次将这三个数对应乘的数比较，取最小的，然后相应的数进入下一轮。

```java
import java.util.*;
public class Solution {
    public int GetUglyNumber_Solution(int index) {
        if (index < 1) return 0;
        int a = 0, b = 0, c = 0;
        List<Integer> list = new ArrayList<Integer>();
        list.add(1);
        while (list.size() < index){
            int temp1 = list.get(a) * 2;
            int temp2 = list.get(b) * 3;
            int temp3 = list.get(c) * 5;
            // 得到 temp1,temp2,temp3中最小的数
            int min = Math.min(temp1,Math.min(temp2,temp3));
            list.add(min);
            // 最小数对应的数进入下一轮
            if(min == temp1) a++;
            if(min == temp2) b++;
            if(min == temp3) c++;
        }
        return list.get(index-1);
    }
}
```

# 34、第一个只出现一次的字符

## 题目描述

> 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.

## 题解

```java
import java.util.*;
public class Solution {
    public int FirstNotRepeatingChar(String str) {
      	// 统计每个字母出现的次数
        Map<Character,Integer> map = new HashMap(26);
        for (char c : str.toCharArray()){
            map.put(c,map.getOrDefault(c,0)+1);
        }
      	// 找到第一个字母出现次数为1的
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (map.get(c) == 1){
                return i;
            }
        }
        return -1;
    }
}
```

# 35、数组中的逆序对

## 题目描述

> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

## 输入描述

> 题目保证输入的数组中没有的相同的数字
>
> 数据范围：	
>
> 对于%50的数据,size<=10^4	
>
> 对于%75的数据,size<=10^5	
>
> 对于%100的数据,size<=2*10^5

**示例1**

输入

> 1,2,3,4,5,6,7,0

输出

> 7

## 题解

使用归并排序的思想，两个排好序的数组，比如2,3,5和1,2,3在比较过程中如果第一个数组的数大于第二个数组的数，那么第一个数组后面的所有数都大于这个数。每两个数组合并成一个数组的时候都可以比较。

```java
public class Solution {
    long res = 0;

    public int InversePairs(int[] array) {
     	 if (array == null || array.length < 2) return 0;
     	 merge(array, 0, array.length - 1);
     	 return (int) (res % 1000000007);
    }

    public void merge(int[] a, int left, int right) {
        if (left < right) {
            int mid = (left + right) / 2;
            merge(a, left, mid);
            merge(a, mid + 1, right);
            mergeSort(a, left, mid, right);
        }
    }

    public void mergeSort(int[] a, int left, int mid, int right) {
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        while (i <= mid && j <= right) {
            res += a[i] > a[j] ? mid - i + 1 : 0;
            temp[k++] = a[i] <= a[j] ? a[i++] : a[j++];
        }
        while (i <= mid) temp[k++] = a[i++];
        while (j <= right) temp[k++] = a[j++];
        for (int t = 0; t < temp.length; t++) a[t + left] = temp[t];
    }
}
```

# 36、两个链表的第一个公共结点

## 题目描述

> 输入两个链表，找出它们的第一个公共结点。

## 题解

**容易理解的方法：**遍历两个链表，计算两个链表的长度，得到两个链表的长度差x。让较长的链表先走x步，此时两个链表距链表尾部的长度一样，两个链表同时遍历，当他们第一次相等时就是第一个公共结点。

```java
public class Solution {
    public ListNode FindFirstCommonNode(ListNode head1, ListNode head2) {
        if (head1 == null || head2 == null) return null;
        int len1=0,len2=0;
        ListNode cur1 = head1;
        ListNode cur2 = head2;
        while (cur1 != null){     // 计算链表1的长度
            len1++;
            cur1 = cur1.next;
        }
        while (cur2 != null){    // 计算链表2的长度
            len2++;
            cur2 = cur2.next;
        }
        int len = Math.abs(len1-len2);  // 两个链表的长度差
        head1 = len1 > len2 ? head1 : head2;  // head1 为较长的
        head2 = len1 > len2 ? head2 : head1;   // head2 为较短的
        while (len>0){    // 较长的链表走长度差 
            len--;
            head1 = head1.next;
        }
        while (head1!=head2){   // 两个链表一起遍历 直到相等
            head1 = head1.next;
            head2 = head2.next;
        }
        return head1;
    }
}
```

**优化后的方法：**两个链表分别记为a和b，链表公共部分长度为z，链表a从头结点到第一个公共结点的距离为x，链表b从头结点到第一个公共结点的距离为y，即链表a的长度为x+z，链表b的长度为y+z。假设链表a的长度小于b，两个链表同时遍历，当a链表遍历结束时，b链表走了x+z，还剩(y+z)-(x+z)=y-x步遍历结束。然后遍历完a链表的结点再遍历一遍b链表，当原遍历b链表的结点遍历完时，再去遍历a链表，此时后遍历b链表的结点距遍历结束还剩(y+z)-(y-x)=z-x步。当后遍历a链表的结点走到第一公共结点的时候，即走了x步，还剩z步。此时后遍历b链表的结点的结点距结束还剩(z-x)+x=z，正好也在第一个公共结点的地方。

```java
public class Solution {
    public ListNode FindFirstCommonNode(ListNode head1, ListNode head2) {
        if (head1 == null || head2 == null) return null;
        ListNode l1 = head1, l2 = head2;
        while (l1 != l2) {
            l1 = l1 == null ? head2 : l1.next;
            l2 = l2 == null ? head1 : l2.next;
        }
        return l1;
    }
}
```

# 37、数字在排序数组中出现的次数

## 题目描述

> 统计一个数字在排序数组中出现的次数。

## 题解

二分，找出这个数第一次和最后一次出现的位置。

```java
public class Solution {
    public int GetNumberOfK(int[] array, int k) {
        if (array == null || array.length == 0) return 0;
        return indexOfRight(array, k) - indexOfLeft(array, k) + 1;
    }

    public int indexOfLeft(int[] a, int k) {
        // 注意这个right不是length-1
        int left = 0, right = a.length;
        // 注意没有等于的情况 因为区间变成了[0，length)而不是[0,length-1]
        while (left < right) {
            int mid = (left + right) / 2;
            if (a[mid] >= k) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    public int indexOfRight(int[] a, int k) {
        int left = 0, right = a.length;
        while (left < right) {
            int mid = (left + right) / 2;
            if (a[mid] > k) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left - 1;
    }
}
```

歪门邪道的方法：找n-0.5和n+0.5的位置

```java
public class Solution {
    public int GetNumberOfK(int [] array , int k) {
        if (array == null || array.length == 0) return 0;
        // 找到k+0.5和k-0.5出现的位置
        return count(array,k+0.5)-count(array,k-0.5);
    }
    public int count(int[] a,double k){
        int left=0,right=a.length-1;
        while(left<=right){
            int mid = (left+right)/2;
            if(a[mid]<k) left = mid+1;
            else right = mid-1;
        }
        return left;
    }
}
```



# 38、二叉树的深度

## 题目描述

> 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

## 题解

```java
public class Solution {
    public int TreeDepth(TreeNode root) {
        if(root==null) return 0;
        return Math.max(TreeDepth(root.left),TreeDepth(root.right))+1;
    }
}
```

# 39、平衡二叉树

## 题目描述

> 输入一棵二叉树，判断该二叉树是否是平衡二叉树。

## 题解

平衡二叉树要求左右孩子结点高度差不超过1

```java
public class Solution {
  
  	boolean isBalance = true;
  
    public boolean IsBalanced_Solution(TreeNode root) {
        if(root==null) return true;
        getDepth(root);
      	return isBalance;
    }
  
  	private int getDepth(TreeNode root){
        if(root==null) return 0;
      	int left = getDepth(root.left);
      	int right = getDepth(root.right);
        if (Math.abs(left-right) > 1){
          	isBalance = false;
        }
        return left >= right ? left+1 : right+1;
    }
}
```

# 40、数组中只出现一次的数字

## 题目描述

> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

## 题解

前提知识

异或，顾名思义，不一样的进行或操作，即0^1=1

+ 两个相等的数字异或结果为0，n^n=0
+ 0和任何数异或结果为任何数，0^n=n

先将数组中所有数字进行异或，最后剩下a和b异或的结果。将这个结果用二进制数看待，找到其中一个为1的位置，数字为1的位置说明原来这两个数在这个位置上肯定一个是0，另一个是1。然后再次遍历数组重新异或，根据这个位置上数的不同，分别异或到两个结果上，原来相同的数字肯定能保证异或到同一个结果上，并消除为0。最后a和b就能区分开。

```java
public class Solution {
    public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
        if (array == null && array.length < 2) return;
        int res = 0;
        // 异或 去除重复的数 得到剩余两个不相同的数的异或结果
        for (int i: array) res ^= i;
        int index = 1;
        // 找到异或结果中第一个1出现的位置
        while ((res&index)==0) index <<= 1;
        // 根据index位置的不同 分别异或
        for (int i : array){
            if ((i&index) == 0) num1[0] ^= i;
            else num2[0] ^= i;
        }
    }
}
```

# 41、和为S的连续正数集合

## 题目描述

> 小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

## 输出描述

> 输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序

## 题解

使用滑动窗口，当窗口内数之和小于给定值时，移动右边界。反之，移动左边界。当一个窗口内数之和正好等于目标数时，将当前窗口内的数保存起来，继续移动右边界找下一组数，直到右边界超出数组范围。

```java
import java.util.ArrayList;
public class Solution {
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> list = new ArrayList();
        int left = 1, right = 2;
        // 减少遍历次数 不用等到right=sum的时候才遍历结束
        while (left < right && right <= sum / 2 + 1) {
            //  当前窗口中的值总和，等差数列求和
            int res = (right + left) * (right - left + 1) / 2;
            if (res < sum) {
                right++;    //当前值小 右边界向右移
            } else if (res > sum){
                left++;//当前值大 左边界向右移
            } else {                   //相等  添加到list中
                ArrayList<Integer> temp = new ArrayList();
                for (int i = left; i <= right; i++){
                    temp.add(i);
                } 
                list.add(temp);
                right++;
            }
        }
        return list;
    }
}
```

# 42、和为S的两个数字

## 题目描述

> 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

## 输出描述

> 对应每个测试案例，输出两个数，小的先输出。

## 题解

根据二次函数数学知识，从数组两边找到和为S的就是乘积最小的。

```java
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
        ArrayList<Integer> list = new ArrayList();
        int i = 0,j = array.length-1;
        while (i < j){
            if (array[i]+array[j] == sum){
                list.add(array[i]);
                list.add(array[j]);
                return list;
            } else if(array[i]+array[j]>sum){
              j--;
            } else{
              i++;
            } 
        }
        return list;
    }
}
```

# 43、左旋转字符串

## 题目描述

> 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

## 题解

**方法一：**简单方法，将字符串分割成两部分然后重新拼接

```java
public class Solution {
    public String LeftRotateString(String str,int n) {
       if (str == null || str.length() == 0) return "";
       return str.substring(n,str.length())+str.substring(0,n);
    }
}
```

**方法二：**将字符串翻转三次，第一次是将整个字符串翻转，第二次将后n位翻转，第三次将0-n的部分翻转。以字符串abcXYZdef为例，第一次翻转后字符串为fedZYXcba，第二次翻转后为fedZYXabc，第三次翻转后为XYZdefabc。代码略。

# 44、翻转单词顺序列

## 题目描述

> 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

## 题解

```java
public class Solution {
    public String ReverseSentence(String str) {
        if(str == null || str.trim().length() == 0) return str;
        String[] strings = str.split(" ");
        StringBuilder sb = new StringBuilder();
        for(int i = strings.length-1;i >= 0;i--){
            if (i == 0) sb.append(strings[i]);
            else sb.append(strings[i]).append(" ");
        }
        return sb.toString();
    }
}
```

# 45、扑克牌顺子

## 题目描述

> LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

## 题解

5张牌，如果有重复的数字，肯定不能组成顺子（大小王除外）。记录5张牌中最小的数和最大的数，0不用统计。如果最后最大数和最小数相差不超过5，说明能组成，否则就不能。

```java
public class Solution {
    public boolean isContinuous(int [] numbers) {
        if (numbers.length != 5) return false;
        int[] a = new int[14];    // 记录每张牌出现的次数
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for(int i:numbers){
            if(i != 0){
                //  遇到重复的牌 肯定不能组成
                if(a[i] != 0) return false;
                // 更新最大最小值
                min = Math.min(min,i);
                max = Math.max(max,i);
            }
            a[i]++;         
        }
        // 到这里没有重复的牌
        return max-min<5;
    }
}
```

# 46、孩子们的游戏(圆圈中最后身下的数)

## 题目描述

> 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
>
> 
>
> 如果没有小朋友，请返回-1

## 题解

**方法一：**使用一个list模拟游戏过程

```java
import java.util.*;
public class Solution {
    public int LastRemaining_Solution(int n, int m) {
        LinkedList<Integer> list = new LinkedList();
        for (int i = 0;i < n;i++) list.add(i);
        int index = 0;
        while (list.size() > 1){
            index = (index+m-1)%list.size();
            list.remove(index);
        }
        return list.size() == 0 ? -1 : list.get(0);
    }
}
```

**方法二：**数学推理，设f(n)表示为最后身下的人

> f(n)：       0,1,2,…m-1,m,m+1,…n-1
>
> f(n-1):      1-m,...       ,n-2,0,1,…,n-m

下面公式由来：原来m-1出列，他后面的m为下一次开始的位置0，他前面的m-2为下一次最后一个n-2，其他位置的数根据这两个顺下去就好。可以得出f(n)=(f(n-1)+m-1)%n，但这不是最后结果，因为这个公式得到的是每个人的编号，而不是第几个人。所以最后公式为f(n)=(f(n-1)+m)%n

```java
public class Solution {
    public int LastRemaining_Solution(int n, int m) {
        if ( n < 2){
            return n-1;
        }
        return (LastRemaining_Solution(n-1,m)+m)%n;
    }
}
```

# 47、求1+2+3+…+n

## 题目描述

> 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

## 题解

等差数列求和 $S=\frac{(1+n)n}{2} =\frac{n^2+n}{2}$

```java
public class Solution {
    public int Sum_Solution(int n) {
        return ((int)Math.pow(n,2) + n) >> 1;
    }
}
```

# 48、不用加减乘除做加法

## 题目描述

> 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

## 题解

二进制加法，每个位上进行异或操作，使用一个变量代表进位，使用与操作。

```java
public class Solution {
    public int Add(int num1,int num2) {
        int sum = num1 ^ num2;    // 两数和  0^1 = 1;1^1 = 0^0=0;
        int carry = num1 & num2;  // 进位    1&1 = 1;1&0 = 0&0 = 0;
        while (carry != 0){
            int temp = sum;           // 保存和
            sum = sum ^ (carry<<1);     // 进位后继续相加
            carry = temp & (carry<<1);  // 相加后的进位
        }
        return sum;
    }
}
```

# 49、把字符串转换成整数

## 题目描述

> 将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0

**输入描述**

> 输入一个字符串,包括数字字母符号,可以为空

**输出描述**

> 如果是合法的数值表达则返回该数字，否则返回0

## 题解

```java
public class Solution {
    public int StrToInt(String str) {
        str = str.trim();       // 去除空格
        char[] c = str.toCharArray();
        if (c==null||c.length<1||(c.length==1&&(c[0]<'0'||c[0]>'9'))) return 0;
        StringBuilder sb = new StringBuilder();
        if ((c[0]>'0'&&c[0]<='9')||c[0]=='-') sb.append(c[0]);
        else if (c[0]!='+')  return 0;      // 第一个字符串不合法
        for (int i = 1; i < c.length; i++) {
            if (c[i]>='0'&&c[i]<='9') sb.append(c[i]);
            else return 0;
        }
        try {
            if (sb.length()==1&&c[0]=='-') return 0;
            return Integer.parseInt(sb.toString());
        }catch (Exception e){
            if (c[0]=='-') return Integer.MIN_VALUE;
            else return Integer.MAX_VALUE;
        }
    }
}
```

# 50、数组中重复的数字

## 题目描述

> 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

## 题解

不使用额外空间，因为数字有范围，我们可以在原数组上修改，将数组上每个位置的值改为对应的位置。为了不污染数组，需要将其他涉及到的值都改了。

```java
public class Solution {
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        for (int i = 0; i < length; i++) {
            // 将i位置上的数变为i
            while (numbers[i] != i) {
                // i位置上的数对应的位置已经为当前数
                if (numbers[numbers[i]] == numbers[i]) {
                    duplication[0] = numbers[i];
                    return true;
                }
                // 将i位置上的数 和 它本应该在的位置上的数 交换
                int temp = numbers[i];
                numbers[i] = numbers[temp];
                numbers[temp] = temp;
            }
        }
        return false;
    }
}
```

# 51、构建乘积数组

## 题目描述

> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

## 题解

B[i]就是数组A中除了i位置，其他位置乘积的结果。遍历两遍，第一遍将[0，i-1]的乘积计算好，B[i]存放着A[0]-A[i-]的乘积。第二遍将[i+1,n-1]的乘积计算好，B[i]存放着A[n-1]-A[i+1]的乘积，再乘上第一遍数组的乘积，就是我们想要的结果。

```java
public class Solution {
    public int[] multiply(int[] A) {
        int[] B = new int[A.length];
        if (B.length == 0) return B;
        B[0] = 1;
        // 前n-1项乘积  每个位置的数为左边的乘积
        for (int i = 1;i < A.length;i++) B[i] = B[i-1]*A[i-1];
        int temp = 1;  // 记录右边的乘积
        // 每个位置的数再乘上右边数的乘积
        for(int i = A.length-2;i >= 0;i--){
            temp *= A[i+1];
            B[i] *= temp; 
        }
        return B;
    }
}
```

# 52、正则表达式匹配

## 题目描述

> 请实现一个函数用来匹配包括'.'和'\*'的正则表达式。模式中的字符'.'表示任意一个字符，而'\*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

## 题解

递归匹配，将待匹配字符串匹配的当前位置记为i，表达式匹配的当前位置记为j。本题重点就是处理`*`，因为它能代表它前一个字符出现的次数，所以我们遍历的时候要判断后一个字符是否为`*`。如果后一个字符为`*`，我们分为两种情况：

+ 当前字符出现0次

  直接将正则表达式匹配位置移到`*`后面的字符，即j+2位置上

+ 当前字符出现大于0次

  先判断当前位置是否能匹配上，如果匹配不上那就不用再往下匹配了。如果匹配上，就将待匹配字符往后继续匹配。

```java
public class Solution {
    public boolean match(char[] str, char[] pattern){
        if (str == null || pattern == null) return false ;
        return match(str,pattern,0,0);
    }
    public boolean match(char[] str,char[] pattern,int i,int j){
        if(j == pattern.length) return i == str.length;
        //  单个字符匹配结果
        boolean match = i < str.length && (pattern[j] == str[i] || pattern[j]=='.');
        //  后面一个为* 两种情况 1：当前字符出现0次  2：当前字符出现n次
        if (j+1 < pattern.length && pattern[j+1] == '*'){
            return match(str,pattern,i,j+2) || (match && match(str,pattern,i+1,j));
        }else{   // 后面不为*  继续匹配
            return match && match(str,pattern,i+1,j+1);
        }
    }
}
```

## 53、表示数值的字符串

## 题目描述

> 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

## 题解

```java
public class Solution {
    public boolean isNumeric(char[] str) {
        String string = String.valueOf(str);
        return string.matches("[\\+\\-]?\\d*(\\.\\d+)?([eE][\\+\\-]?\\d+)?");
    }
}
```

# 54、字符流中第一个不重复的字符

## 题目描述

> 请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

## 输出描述

> 如果当前字符流没有存在出现一次的字符，返回#字符。

## 题解

```java
import java.util.*;
public class Solution {
    Queue<Character> list = new LinkedList();
    HashSet<Character> set = new HashSet();
    //Insert one char from stringstream
    public void Insert(char ch){
        if(set.contains(ch)) list.remove(ch);
        else {
            list.offer(ch);
            set.add(ch);
        }
    }
  //return the first appearence once char in current stringstream
    public char FirstAppearingOnce(){
        if(list.isEmpty()) return '#';
        return list.peek();
    }
}
```

# 55、链表中环的入口结点

## 题目描述

> 给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

## 题解

快慢指针，快指针每次走两步，慢指针每次走一步。快慢指针在环中一定会相遇，相遇的时候快指针从链表头部重新遍历，每次走一步，当快慢指针再次相遇的时候一定为环的入口。

**证明：**设链表头部到环的入口结点长度为a，环的入口结点到第一次相遇的的点距离为b，相遇点再到环的入口结点距离为c，b+c就是环的长度。第一次相遇时，快指针走的距离为a+x(b+c)+b,x为走的圈数，慢指针走的距离为a+y(b+c)+b，y为走的圈数。两者时间一样，快指针速度为慢指针的两倍，所以距离为两倍关系，得2(a+y(b+c)+b)=a+x(b+c)+b，化简一下。a=(x-2y)(b+c)-b，再好看一点，a=(x-2y-1)(b+c)+c。快指针从链表头到入口结点的距离为a，慢指针距入口结点距离为c，因为两者速度一样，所以当慢指针再走x-2y-1圈+c时，正好相遇。

```java
public class Solution {

    public ListNode EntryNodeOfLoop(ListNode head){
        if(head==null) return null;
        ListNode fast = head;     // 快指针
        ListNode slow = head;     // 慢指针
        while (fast != null && fast.next != null){
            fast = fast.next.next;  // 快指针每次走两步
            slow = slow.next;       // 慢指针每次走一步
            if(fast == slow){       // 快慢指针相等  快指针回到头结点
                fast = head;
                while(fast!=slow){   // 快指针和慢指针相遇的点就为环的入口结点
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }
}
```

# 56、删除链表中重复的节点

## 题目描述

> 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

## 题解

```java
public class Solution {
    public ListNode deleteDuplication(ListNode head){
        if (head == null) return null;
        ListNode cur = head;
        ListNode pre = null;
        ListNode temp = null;
        while (cur != null){
            temp = cur.next;  // 向后遍历的节点
            while(temp != null && temp.val == cur.val) temp = temp.next;
            if (temp != cur.next){   // 有重复节点
                if (pre == null)  head = temp;   // 头节点重复  更换头结点 
                else pre.next = temp;         // 其他节点重复 去除重复节点
                cur = temp;
            }else{             // 没有重复节点
                 pre = cur;
                 cur = cur.next;
            }
        }
        return head;
    }
}
```

# 57、二叉树的下一个结点

## 题目描述

> 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

## 题解

分几种情况：

+ 当前结点是叶子结点

  + 当前节点是左叶子结点

    中序遍历的下一个结点就是它的父结点

  + 当前节点是右叶子结点

    递归寻找

+ 当前结点不是叶子结点

  + 有右孩子结点

    返回右孩子结点的最左叶子结点

  + 没有右孩子结点

    返回父结点

```java
public class Solution {
    public TreeLinkNode GetNext(TreeLinkNode root){
        if (root.right != null){    // 当前节点存在右孩子
            root = root.right;   // 找到右孩子的最左叶子节点
            while (root.left!=null) root = root.left;
            return root;
        }else {                  // 当前节点不存在右孩子
            TreeLinkNode parent = root.next; // 当前节点是左孩子 直接返回父节点
            while (parent != null && parent.left != root){ // 当前节点是右孩子 返回父节点的父节点
                root = parent;
                parent = parent.next;
            }
            return parent;
        }
    }
}
```

# 58、对称的二叉树

## 题目描述

> 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

## 题解

```java
public class Solution {
  
    boolean isSymmetrical(TreeNode root){
        return isMirror(root,root);
    }
  
    private boolean isMirror(TreeNode left,TreeNode right){
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        return isMirror(left.left,right.right) && isMirror(left.right,right.left);
    }
}
```

# 59、按之字形顺序打印二叉树

## 题目描述

> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

## 题解

```java
public class Solution {
    public ArrayList<ArrayList<Integer>> Print(TreeNode root) {
        ArrayList<ArrayList<Integer>> list = new ArrayList();
        if (root == null) return list;
        LinkedList<TreeNode> queue = new LinkedList();
        queue.offer(root);
        int level = 1;
        while(!queue.isEmpty()){     // 二叉树的层次遍历
            int size = queue.size();
            ArrayList<Integer> temp = new ArrayList();
            for(int i = 0;i < size;i++){
                root = queue.poll();
                if((level&1) == 1) temp.add(root.val);
                else temp.add(0,root.val);     //从后往前插入
                if(root.left != null) queue.offer(root.left);
                if(root.right != null) queue.offer(root.right);
            }
            level++;
            list.add(temp);
        }
        return list;
    }
}
```

# 60、把二叉树打印成多行

## 题目描述

> 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

## 题解

```java
public class Solution {
    ArrayList<ArrayList<Integer> > Print(TreeNode root) {
        ArrayList<ArrayList<Integer>> list = new ArrayList();
        if (root == null) return list;
        Queue<TreeNode> queue = new LinkedList();
        queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            ArrayList<Integer> temp = new ArrayList();
            for(int i = 0;i < size;i++){
                root = queue.poll();
                temp.add(root.val);
                if(root.left != null) queue.offer(root.left);
                if(root.right != null) queue.offer(root.right);
            }
            list.add(temp);
        }
        return list;
    } 
}
```

# 61、序列化二叉树

## 题目描述

> 请实现两个函数，分别用来序列化和反序列化二叉树
>
> 
>
> 二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。
>
> 二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

## 题解

选择一种遍历方式序列化，空节点用一个特殊字符表示，每个节点间再用另一种字符分隔

```java
import java.util.*;
public class Solution {
    // 先序遍历序列化二叉树 空节点用#表示 每个节点用，隔开
    String Serialize(TreeNode root) {
        if(root == null) return "#,";
        String str = root.val + ",";
        str += Serialize(root.left);
        str += Serialize(root.right);
        return str;
    }
    // 使用队列先序遍历反序列化二叉树
    TreeNode Deserialize(String str) {
        String[] strs = str.split(",");
        Queue<String> queue = new LinkedList();
        for(String s : strs) queue.offer(s);
        return Deserialize(queue);
    }
    TreeNode Deserialize(Queue<String> queue) {
        String str = queue.poll();
        if(str.equals("#")) return null;
        TreeNode root = new TreeNode(Integer.parseInt(str));
        root.left = Deserialize(queue);
        root.right = Deserialize(queue);
        return root;
    }
}
```

# 62、二叉搜索树的第k个结点

## 题目描述

> 给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

## 题解

二叉搜索树需要使用中序遍历

```java
import java.util.*;
public class Solution {
    // 二叉树的中序遍历 非递归
    TreeNode KthNode(TreeNode root, int k){
        Stack<TreeNode> stack = new Stack();
        int count = 1;
        while (root != null || !stack.isEmpty()){
            if (root != null){
                stack.push(root);
                root = root.left;
            }else{
                root = stack.pop();
                if(k == count++) return root;
                root = root.right;
            }
        }
        return null;
    }
}
```

# 63、数据流中的中位数

## 题目描述

> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

## 题解

使用一个小顶堆和一个大顶堆，两个堆中的元素差不能超过1。当堆中元素不符合要求时，调整两个堆中元素数量试其满足要求。

```java
import java.util.*;
public class Solution {
    
    PriorityQueue<Integer> minHeap = new PriorityQueue();
    PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>((o1,o2) -> o2-o1);

    public void Insert(Integer num) {
        if(maxHeap.isEmpty()||num<=maxHeap.peek()) maxHeap.offer(num);
        else minHeap.offer(num);
        if(maxHeap.size()-minHeap.size()>1) minHeap.offer(maxHeap.poll());
        else if(minHeap.size()-maxHeap.size()>1) maxHeap.offer(minHeap.poll());
    }

    public Double GetMedian() {
        if(minHeap.size()>maxHeap.size()) return new Double(minHeap.peek());
        else if(minHeap.size()<maxHeap.size()) return new Double(maxHeap.peek());
        else return (minHeap.peek()+maxHeap.peek())/2.0;
    }

}
```

# 64、滑动窗口的最大值

## 题目描述

> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

## 题解

```java
import java.util.*;

public class Solution {
    public ArrayList<Integer> maxInWindows(int [] nums, int size){
        ArrayList<Integer> list = new ArrayList();
        if(nums.length<size || nums==null|| nums.length<1 || size<1) return list;
        // 双端队列 保存当前窗口最大值对应的数组下标
        LinkedList<Integer> queue = new LinkedList();
        for(int i = 0;i < nums.length;i++){  // 遍历数组
            // 保证队列中对应数组的数 从大到小
            while(!queue.isEmpty() && nums[i] >= nums[queue.peekLast()]){
                queue.pollLast();
            }
            queue.offer(i);
            // 窗口形成 并且 当前窗口中的最大值已失效
            if(i-size >= queue.peek()) queue.poll();
            // 窗口形成 保存窗口最大值
            if(i >= size-1) list.add(nums[queue.peek()]);
        }
        return list;
    }
}
```

# 65、矩阵中的路径

## 题目描述

> 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如 $\begin{bmatrix} a & b & c &e \\ s & f & c & s \\ a & d & e& e\\ \end{bmatrix}\quad$ 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

## 题解

```java
public class Solution {
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str){
        if(matrix==null||matrix.length==0) return false;
        boolean[] flag = new boolean[matrix.length];
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                if(dfs(matrix,flag,i,j,rows,cols,str,0)) return true;
            }
        }
        return false;
    }
    public boolean dfs(char[] matrix,boolean[] flag,int i,int j,int row,int col,char[] str,int index){
        int k = i*col+j;
        if(i<0||i>=row||j<0||j>=col||flag[k]==true||str[index]!=matrix[k]||index>str.length) return false;
        if(index==str.length-1) return true;
        flag[k] = true;
        boolean res = dfs(matrix,flag,i+1,j,row,col,str,index+1)||
            dfs(matrix,flag,i-1,j,row,col,str,index+1)||
            dfs(matrix,flag,i,j+1,row,col,str,index+1)||
            dfs(matrix,flag,i,j-1,row,col,str,index+1);
        flag[k] = false;
        return res;
    }
}
```

# 66、机器人的运动范围

## 题目描述

> 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

## 题解

```java
public class Solution {

    public int movingCount(int threshold, int rows, int cols){
        int[][] flag = new int[rows][cols];
        return dfs(flag,0,0,rows,cols,0,threshold);
    }
    public int dfs(int[][] flag,int i,int j,int row,int col,int count,int k){
        if(i<0||i>=row||j<0||j>=col||flag[i][j]==1||sum(i,j)>k) return 0;
        flag[i][j] = 1;
        return dfs(flag,i-1,j,row,col,count+1,k)+
        dfs(flag,i+1,j,row,col,count+1,k)+
        dfs(flag,i,j-1,row,col,count+1,k)+
        dfs(flag,i,j+1,row,col,count+1,k)+1;
    }
    public int sum(int i,int j){
        int sum = 0;
        while(i>0){
            sum+=i%10;
            i/=10;
        }
        while(j>0){
            sum+=j%10;
            j/=10;
        }
        return sum;
    }
}
```

# 67、剪绳子

## 题目描述

> 给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

## 题解

```java
public class Solution {
    public int cutRope(int target) {
        int[] dp = new int[target+1];
        dp[1] = 1;
        for(int i = 2;i <= target;i++){
            for(int j = 1;j < i;j++){
                dp[i] = Math.max(dp[j]*dp[i-j],dp[i]);
                dp[i] = Math.max(j*dp[i-j],dp[i]);
                dp[i] = Math.max(j*(i-j),dp[i]);
            }
        }
        return dp[target];
    }
}
```

