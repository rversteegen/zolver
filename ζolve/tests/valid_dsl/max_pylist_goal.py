x : Int
y : Real
constraint(x <= 1)
constraint(y <= 1 + x)
goal = max([x, y])
expected_answer = 2
