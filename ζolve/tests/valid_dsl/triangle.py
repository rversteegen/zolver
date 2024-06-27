#precalculus-1271
#In triangle $ABC,$ we have $\angle C = 3\angle A,$ $a = 27,$ and $c = 48.$  What is $b$?
#Note: $a$ is the side length opposite $\angle A,$ etc. 

## TRANS
# Define the variables for the angles measured in degrees
A, B, C : Real
# The given relationship between the angles
C == 3 * A
# The sum of the angles in a triangle is 180 degrees
sum(A, B, C) == 180
# Define the side lengths
a, b, c : Real
a == 27
c == 48
# The Law of Sines
sin(A) / a == sin(B) / b == sin(C) / c
# Find b
goal = b
