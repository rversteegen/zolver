###DeepSeekMath-base_v8.1_MATH_train_translations.csv int_algebra-505
## PROBLEM
#Let $P(x) = 0$ be the polynomial equation of least possible degree, with rational coefficients, having $\sqrt[3]{7} + \sqrt[3]{49}$ as a root.  Compute the product of all of the roots of $P(x) = 0.$ 

## TRANS
# Let the unknown polynomial be:
p : Function((Real), Real)
# The given root is a single argument variable
r : Real
r == (7^(1/3) + 49^(1/3))
# The root is a zero of the polynomial:
p(r) == 0
# The degree of the polynomial is minimized, so all other roots are the conjugates (negative and reciprocal) of the given root:
p(r^2) == 0
p(1/r) == 0
p(-r) == 0
p(-r^2) == 0
p(1/r^2) == 0
# The product of all the roots is the constant term of the polynomial
goal = p(0)
expected_answer = 56
