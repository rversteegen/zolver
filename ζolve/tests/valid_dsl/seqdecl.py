
# Jen picks 4 numbers from the set S
Jen : Seq(Int, len = 4, unique = True)
# Let the randomly chosen numbers be r_1, r_2, r_3, and r_4
r : Seq(Int, len = 4, unique = True)
# Jen wins a prize if at least 2 of her numbers are in the randomly chosen numbers
ForAll(2 in count(set(Jen) & set(r)) for r in Seq(Int, len = 4, unique = True))
# Jen wins the grand prize if all 4 of her numbers are in the randomly chosen numbers
And(Jen in set(r) for r in Seq(Int, len = 4, unique = True))
# The probability of winning the grand prize given that Jen won a prize is:
goal = count(Jen in set(r) for r in Seq(Int, len = 4, unique = True)) / count(ForAll(2 in count(set(Jen) & set(r)) for r in Seq(Int, len = 4, unique = True)))
