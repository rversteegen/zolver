b: bool
c = divides(4, 16)
Iff(c, b)
goal = If(If(b, Not(b), True), 4, 5)   # equiv to If(Not(b), 4, 5)
expected_answer = 5 
