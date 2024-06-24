###### Problem:
#What is the value of $n$ in the equation $n + (n + 1) + (n + 2) = 9$?

x : Real
y : Real
z : Complex
constraint(x + (x+1) + (x+2) == 9)
# Oddly, z3 returns sat:
# defaultopt check() = sat in 0.9ms
# z3 model() = [y = 0, x = 2]
# so min(x + y)  evaluated to  2
# solve(): objective range is  -1*oo to -1*oo

goal = min(x + y)

expected_answer = -oo
