###### Problem counting_prob-734:
#Given the equation $a + b = 30$, where $a$ and $b$ are positive integers, how many distinct ordered-pair solutions $(a, b)$ exist?

###### Translation:
# a and b are positive integers, so we declare them as such
a, b : Int
a > 0
b > 0
# The equation a + b = 30 is given
a + b == 30
# The number of possible solutions is the number of positive integer pairs (a, b) that satisfy the equation
goal = count((a, b) for a in Int for b in Int if a > 0 and b > 0 and a + b == 30)
