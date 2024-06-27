s : Seq(Int, len = 4)
s[0] == 0
s[1] == 5
s[2] == 10
average(s) == 6  # s[3] == 9
# Test for Contains too
goal = max(x^2 for x in s if x < 10)
expected_answer = 81
