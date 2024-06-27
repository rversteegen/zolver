s : Seq(Int, len = 5)
# The median of s is 9
#median(s) == 9
# The average of s is 10
average(s) == 10
# There is only one 8 in s
#count(x == 8 for x in s) == 1
# The largest possible integer that could appear in s is the maximum of all possible assignments to s
#goal = max(s)
s[0]==2
goal = median(s)
expected_answer = 'notunique'
