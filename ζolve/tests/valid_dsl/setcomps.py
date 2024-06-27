# ## PROBLEM
# A list of five positive integers has all of the following properties:

# $\bullet$  The only integer in the list that occurs more than once is $8,$

# $\bullet$  its median is $9,$ and

# $\bullet$  its average (mean) is $10.$

# What is the largest possible integer that could appear in the list? 

## TRANS
# Declare the unknown list s as a sequence of integers
s : Seq(Int, len = 5)
# The median of s is 9
median(s) == 9
# The average of s is 10
average(s) == 10
# There is only one 8 in s
count(x == 8 for x in s) == 1
# The largest possible integer that could appear in s is the maximum of all possible assignments to s
goal = min(s)
