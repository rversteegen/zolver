# ## PROBLEM
# A list of five positive integers has all of the following properties:

# $\bullet$  The only integer in the list that occurs more than once is $8,$

# $\bullet$  its median is $9,$ and

# $\bullet$  its average (mean) is $10.$

# What is the largest possible integer that could appear in the list? 

## TRANS
# Name the list x
x : Seq(Int, len = 5)
# 8 occurs more than once
ForAll(x[i] == 8 for i in Int if i != j and x[i] == x[j] and j in Int)
# The median is 9
median(x) == 9
# The average is 10
average(x) == 10
# The largest possible integer is the maximum of x
goal = max(x)
expected_answer = '?'
