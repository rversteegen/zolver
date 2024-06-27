# Merge14Mistral7b_v8.1_MATH_train_translations.csv algebra-2414
## PROBLEM
#For how many positive integers $n$ does $1+2+\cdots+n$ evenly divide $6n$? 

## TRANS
# Sum of the first n natural numbers

sum_n = sum(range(1, 10 + 1))  #sum(1, n + 1)
# The goal is to find n for which sum_n divides 6n
# We can use the mod operator to find the remainder when 6n is divided by sum_n
remainder = 6*n % sum_n
# We want the remainder to be 0
remainder == 0
# Find the number of such n
goal = count(n)
expected_answer = '?'
