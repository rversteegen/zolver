
## TRANS
# Define the arithmetic sequence as an unknown function a: Int -> Real
a : Function((Int), Real)
# The first term is unknown, but the difference is known
#a[0] : Real
d : Real
ForAll(a[i] == a[0] + i * d for i in Int)
# The 16th and 17th terms of the sequence are 8 and 10
a[15] == 8
a[16] == 10
# Find the second term
goal = a[1]
