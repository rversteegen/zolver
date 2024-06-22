x : Int
y : Real
constraint(y == x + 1)
goal = min(x**2, (y + 0.5)**2 - 0.5)
expected_answer = -0.25   # eg x = -1, y = 0
