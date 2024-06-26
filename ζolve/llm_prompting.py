import time
from collections import defaultdict
from ζ import dsl, dsl_parse, ζ3, solver


def build_prompt(filename = "prompt.txt", sections = ['seqs', 'graphs', 'complex', 'ntheory'], startcode = '```', endcode = '```', maxlength = 10000):

    startanswer = "### CAS Translation\n" + startcode

    with open(filename) as pfile:
        lines = pfile.read()

    selected = ""
    including = True
    for line in lines.split("\n"):
        if line.startswith('[NOTE]'):
            continue
        if line.startswith('[ELSE]'):
            including = not including
            continue
        if line.startswith('[BLOCK]'):
            if len(selected) > maxlength:
                print("build_prompt: Reached maxlength!!")
                break
            keys = line.split(" ")[1:]
            if 'or' in keys:
                keys.remove('or')
                including = any(k in sections for k in keys)
            else:
                including = all(k in sections for k in keys)
        elif including:
            selected += line + "\n"

    if False:
        # Remove comments
        lines = []
        for line in selected.split("\n"):
            if line.startswith('# '): continue
            lines.append(line)
        selected = '\n'.join(lines)

    prompt = selected
    prompt = prompt.replace('STARTCODE', startcode).replace('ENDCODE', endcode).replace("STARTANSWER", startanswer)
    return prompt.rstrip() + "\n"

#print(build_prompt("prompt.txt"))

def topic_query(gen, problem):
    "gen is a fresh LLMGenerator"


    prompt = r"""# User instruction
Categorise maths problems by topic:

# Problem
"What is the sum of all values of $y$ for which the complex function $\frac{y+6}{y^2-5y+4}$ is undefined?"

# Topics
Graphs of functions: Yes
Combinatorics: No
Statistics: No
Number theory: No
Complex numbers: Yes
Vectors and matrices: No
Recurrence relations: No
Sequences: No
Geometry: No

# Problem
"Let $S$ be a region in the plane with area 10.  When we apply the matrix \[\begin{pmatrix} 2 & 1 \\ 7 & -3 \end{pmatrix}\]to $S,$ we obtain the region $S'.$  Find the area of $S'.$"

# Topics
Graphs of functions: No
Combinatorics: No
Statistics: No
Number theory: No
Complex numbers: No
Vectors and matrices: Yes
Recurrence relations: No
Sequences: No
Geometry: Yes

# Problem
"What is 9876 * 10?"

# Topics
Graphs of functions: No
Combinatorics: No
Statistics: No
Number theory: No
Complex numbers: No
Vectors and matrices: No
Recurrence relations: No
Sequences: No
Geometry: No

# Problem
"The Smith family has 4 sons and 3 daughters. In how many ways can they be seated in a row of 7 chairs such that no boys are next to each other?"

# Topics
Graphs of functions: No
Combinatorics: Yes
Statistics: No
Number theory: No
Complex numbers: No
Vectors and matrices: No
Recurrence relations: No
Sequences: Yes
Geometry: No

# Problem
"Find the smallest four-digit palindrome which is the cube of a prime."

# Topics
Graphs of functions: No
Combinatorics: No
Statistics: No
Number theory: Yes
Complex numbers: No
Vectors and matrices: No
Recurrence relations: No
Sequences: Yes
Geometry: No

# Problem
"PROBLEM"

# Topics"""


    topics = """graphs|Graphs of functions:
comb|Combinatorics:
stats|Statistics:
ntheory|Number theory:
complex|Complex numbers:
linalg|Vectors and matrices:
relations|Recurrence relations:
seqs|Sequences:
geometry|Geometry:""".split("\n")

    gen.append_prompt(prompt.replace('PROBLEM', problem), show=False)

    sections = []
    for topic in topics:
        code, name = topic.split("|")
        gen.append_prompt("\n" + name, show=False)
        gen.generate(0.1, top_p = 0.1, limit = 1, show=False)
        if gen.new_output == " Yes":
            sections.append(code)
        if gen.new_output not in (" Yes", " No"):
            print("WARNING: UNRECOGNISED TOPIC ANSWER:", gen.new_output)


    print("Topics:", sections)
    return sections

def choose_prompt(gen, problem, maxlength = 10000):

    topics = topic_query(gen, problem)
    if len(topics) == 0:
        # Very easy calculation problem
        pass #topics = ['seqs', 'graphs', 'ntheory']

    prompt = build_prompt("/kaggle/input/zolver/ζolve/prompt.txt", topics, maxlength = maxlength)
    return prompt.replace('PROBLEM', problem)


class ScoreLog:
    def __init__(self):
        self.outputs = []  # List of (answer, score, info) tuples
        self.answer_scores = defaultdict(int)  # answer -> total_score

    def add_answer(self, answer, score, result_info):
        print(f"RESULT = {answer} SCORE = {score}")
        answer = int(round(answer)) % 1000
        self.outputs.append((answer, score, result_info))
        self.answer_scores[answer] += max(0, score)

        if len(self.outputs) > 0:
            answers = [(score,ans) for (ans,score) in self.answer_scores.items()]
            answers.sort(reverse = True)
            print("SCORES,ANSWERS:", answers)
            best_score, best = answers[0]
            if len(answers) >= 2:
                score_gap = best_score - answers[1][0]
            else:
                score_gap = best_score
            return best, best_score, score_gap


class ζolver:
    def __init__(self, problem, multiprocessing = None, maxlength = 10000):
        self.maxlength = maxlength
        self.problem = problem
        self.scorelog = ScoreLog()
        self.best = 0
        self.multiprocessing = multiprocessing

    def doit(self, makegen, timeout = 180, hard_timelimit = None):
        """
        Returns true if definitively solved, otherwise rsulsts in scorelog.
        hard_timelimit: timestamp must finish before
        """
        start_time = time.time()

        print("###STATEMENT\n" + self.problem + "\n")

        prompt = choose_prompt(makegen(), self.problem, self.maxlength)

        temp = 0.3

        REPEATS = 6

        for repeat in range(REPEATS):
            it_start = time.time()
            if hard_timelimit and it_start > hard_timelimit:
                return

            time_left = min(start_time + timeout - it_start, hard_timelimit - time.time())
            if time_left < 5:
                return
            # Give each repeat at least timeout/REPEATS time
            #time_for_item = min(time_left, timeout / max(1, REPEATS - repeat)
            time_for_item = max(10, min(time_left,  30))

            try:
                gen = makegen()
                gen.append_prompt(prompt, show = False)

                temp = max(0.9, 0.3 + repeat * 0.2)
                gen.generate(temp, limit = 600, skip_check = True)
                translation = gen.new_output

            except Exception as e:
                # Could be a OOM
                print("gen.generate() EXCEPTION")
                print(e)
                continue

            try:
                workspace = dsl_parse.load_dsl(translation, verbose = False)
                print("PARSE SUCCESS")
                workspace.print()

                res = workspace.solve()

                print("---------ζOLVE RESULT")
                print(f"Result {res}, answer {workspace.solution}")
                #if res == solver.unknown orres == solver.unsat or res == solver.notunique:
                if not solver.solved:
                    print("No conclusion.")
                    continue

                score = 1
                info = "ζolve"
                if workspace.goal.is_constant:
                    score = 0.6
                    info = "ζolve-constant"
                self.best, best_score, score_gap = self.scorelog.add_answer(solver.solution, score, info)
                if (score_gap >= 2 or best_score >= 1.7): # and not VALIDATE:
                    print("ζOLVE EARLY FINISH")
                    return
                    #return True  # FIXME

            except NotImplementedError as e:
                print("-------ζOLVE FAILED: NotImplementedError")
                print(e)
                #stats.unimp += 1
                #stats.solvefailed += 1
            except (SyntaxError, dsl.DSLError, ζ3.MalformedError) as e:
                print("-------ζOLVE FAILED")
                print(e)
                #stats.solvefailed += 1
            except Exception as e:
                # Could be a OOM
                print("-------ζOLVE UNCAUGHT EXCEPTION")
                print(e)


if __name__ == '__main__':
    ζol = ζolver("foo")
