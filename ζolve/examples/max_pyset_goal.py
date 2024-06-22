x : Real
y : Int
constraint(x <= 1)
constraint(y <= 2.9 + x)
goal = max({x, y})
expected_answer = 3
