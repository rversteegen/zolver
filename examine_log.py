#!/usr/bin/env python3

"""
Compare average scores, times, code_blocks, etc for different prompts from
data from logs/*.csv files.
"""

from collections import defaultdict
import pandas as pd

prob = pd.read_csv("data/ArtOfProblemSolving/AMC_12_valid_old.csv", index_col = 0)
prob = prob.drop(columns=['link', 'problem', 'letter', 'id', 'competition'])
prob = prob.rename({'answer':'solution'}, axis = 1)

# AMC12o means AMC_12_valid_old.csv
# AMC12V means AMC_12_valid.csv

if True:
    sol = pd.read_csv("logs/20240529-0115-AMC12o_2xT4.csv", index_col = 0)
    sol['tag'] = '529'
    sol2 = pd.read_csv("logs/20240529-1355-70ofAMC12o_Qoff_1500-700tok_P100.csv", index_col = 0)
    sol2['tag'] = '529'
    sol3 = pd.read_csv("logs/20240608-1610-30ofAMC12o_Qoff_2048-2048tok_2xT4.csv", index_col = 0)
    sol3['tag'] = '608'

    sol = pd.concat([sol,sol2,sol3])
elif False:
    sol = pd.read_csv("logs/20240608-1227-30ofAMC12V_Qoff_2048-2048tok_P100_noTB_.csv", index_col = 0)  # Not actually noTB?
    prob = pd.read_csv("data/ArtOfProblemSolving/AMC_12_valid.csv", index_col = 0)
else:
    sol = pd.read_csv("logs/20240608-1610-30ofAMC12o_Qoff_2048-2048tok_2xT4.csv", index_col = 0)
    sol['tag'] = '608'


sol = sol.rename({'problem_id':'prob_name'}, axis = 1)


com = sol.merge(prob, on='prob_name')
com['correct'] = com.answer == com.solution
com['scored'] = com.correct * com.score

prob.set_index('prob_name', inplace = True, drop = False)

print("probs", len(pd.DataFrame(com.value_counts('prob_name'))))

prompts = pd.DataFrame(com.value_counts(['tag','prompt']))

# For each of the prompt/tag combinations
for p in prompts.index:
    tag, prompt = p
    thisp = com[(com.tag == tag) & (com.prompt == prompt)]
    prompts.loc[p, 'avgtime'] = thisp.time.mean()
    prompts.loc[p, 'correct'] = thisp.correct.mean()
    prompts.loc[p, 'score'] = thisp.score.mean()
    prompts.loc[p, 'scored'] = thisp.scored.mean()
    prompts.loc[p, 'gen_tokens'] = thisp.gen_tokens.mean()
    prompts.loc[p, 'code_blocks'] = thisp.code_blocks.mean()
    prompts.loc[p, 'code_errors'] = thisp.code_errors.mean()
prompts['errorrate'] = prompts['code_errors'] / prompts['code_blocks']

prob['answers'] = [defaultdict(float) for i in range(len(prob))]
prob['scores'] = [defaultdict(float) for i in range(len(prob))]
prob['right_score'] = 0.0
prob['wrong_score'] = 0.0
prob['right_ans'] = 0
prob['wrong_ans'] = 0
prob['solved'] = 0.0  # 1 if solved by maximum score, fractional if a tie
prob['solvedcount'] = 0.0  # 1 if solved by majority vote, fractional if a tie
prob['solvedfrac'] = 0.0  # fraction of total score assigned to the right answer
# For each attempt at a solution, tally it onto its problem
for _, soln in com.iterrows():
    #thisprob = prob.loc[soln.prob_name]
    i = soln.prob_name
    if soln.correct:
        prob.loc[i, 'right_score'] += soln.score
        prob.loc[i, 'right_ans'] += 1
    else:
        prob.loc[i, 'wrong_score'] += soln.score
        prob.loc[i, 'wrong_ans'] += 1
    if soln.answer > -1:
        prob.loc[i, 'answers'][soln.answer] += 1
        prob.loc[i, 'scores'][soln.answer] += soln.score
prob['solvedfrac'] = prob.right_score / (prob.right_score + prob.wrong_score)

# But is it the most common? Tally up the scores for each solution
for i, p in prob.iterrows():
    if len(p.scores):
        def compute_solved(ans_counts):
            answers = [(score,ans) for (ans,score) in ans_counts]
            answers.sort(reverse = True)
            best_score, best = answers[0]
            mix = [ans for (score,ans) in answers if score == best_score]
            return int(p.solution in mix) / len(mix)
        prob.loc[i,'solved'] = compute_solved(p.scores.items())
        prob.loc[i,'solvedcount'] = compute_solved(p.answers.items())
    #else:
    #    prob = prob.drop(p)

# for _, p in prob.iterrows():
#     if len(p.scores):

# Drop probs with no solutions
prob = prob.drop(filter((lambda p: len(prob.loc[p].scores) == 0), prob.index))

prob = prob.drop(columns = ['answers', 'scores', 'prob_name'])

#print(len(
print(prompts)

print(prob)

print(f"solved = {prob.solved.mean() * 100 :.1f}%")
print(f"solvedcount = {prob.solvedcount.mean() * 100 :.1f}%")
print(f"solvedfrac = {prob.solvedfrac.mean() * 100 :.1f}%")
