
import time
from collections import defaultdict
import ζ.solver, ζ.dsl, ζ.dsl_parse, ζ.ζ3
#from ζ import dsl, dsl_parse, ζ3

################################################################################
## Prompt and topic

PROMPT_FILE = "/kaggle/input/zolver/ζolve/prompt.txt"  #temp

def build_prompt(filename = "prompt.txt", sections = ['seqs', 'graphs', 'complex', 'ntheory'], startcode = '```', endcode = '```', maxlength = 10000):
    print(sections)
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

    if len(selected) > maxlength:
        print("build_prompt: Reached maxlength!!", len(selected), ">", maxlength)
        sections.pop()
        return build_prompt(filename, sections, startcode, endcode, maxlength)

    prompt = selected
    prompt = prompt.replace('STARTCODE', startcode).replace('ENDCODE', endcode).replace("STARTANSWER", startanswer)
    return prompt.rstrip() + "\n"


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

    prompt = build_prompt(PROMPT_FILE, topics, maxlength = maxlength)
    return prompt.replace('PROBLEM', problem)

################################################################################
## LLM

try:
    from llm_util import run_llm
except ImportError:

    def run_llm(prompt, *args, numseqs = 1, **_kwargs):
        print("seqs", numseqs)
        return ["""
x : Int
constraint(is_prime(x))
constraint(x >= 98)
goal = min(x)"""] * numseqs

    


################################################################################
## 

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

    def doit(self, model, tokenizer, makegen, timeout = 180, hard_timelimit = None):
        """
        Returns true if definitively solved, otherwise rsulsts in scorelog.
        hard_timelimit: timestamp must finish before
        """
        start_time = time.time()
        if hard_timelimit is None:
            hard_timelimit = start_time + 1e6

        def time_left():
            return min(start_time + timeout, hard_timelimit) - time.time()

        print("###STATEMENT\n" + self.problem + "\n")

        prompt = choose_prompt(makegen(), self.problem, self.maxlength)

        REPEATS = 4

        for repeat in range(REPEATS):
            print(repeat, "timeleft", time_left())
            if time_left() < 30:
                return
            # Give each repeat at least timeout/REPEATS time
            #time_for_item = min(time_left, timeout / max(1, REPEATS - repeat)
            #time_for_item = max(10, min(time_left(),  30))

            try:
                # gen = makegen()
                # gen.append_prompt(prompt, show = False)

                temp = max(0.9, 0.4 + repeat * 0.15)
                # gen.generate(temp, limit = 600, skip_check = True)
                # outputs = [gen.new_output]

                outputs = run_llm(model, tokenizer, prompt, max_tokens = 640, numseqs = 3, temp = temp, stopwords = ["```"])
            except Exception as e:
                # Could be a OOM
                print("gen.generate() EXCEPTION")
                print(e)
                continue

            for translation in outputs:
                print("-------Translation--------")
                print(translation)
                print("--------------------------")

                if time_left() < 5:
                    return
                if self.try_translation(translation):
                    return True  # FIXME

    def try_translation(self, translation):
        try:
            workspace = ζ.dsl_parse.load_dsl(translation, verbose = False) # A ζ.solver.Workspace
            print("PARSE SUCCESS")
            workspace.print()

            res = workspace.solve()

            print("---------ζOLVE RESULT")
            print(f"Result {res}, answer {workspace.solution}")
            #if res == ζ.solver.unknown orres == ζ.solver.unsat or res == ζ.solver.notunique:
            if res != ζ.solver.solved:
                print("No conclusion.")
                return

            score = 2.
            info = "ζolve"
            if ζ.dsl.is_a_constant(workspace.goal):
                score = 0.8
                info = "ζolve-constant"
            self.best, best_score, score_gap = self.scorelog.add_answer(workspace.solution, score, info)
            if (score_gap >= 6 or best_score >= 7): # and not VALIDATE:  ####FIXM
                print("ζOLVE EARLY FINISH")
                return True

        except NotImplementedError as e:
            print("-------ζOLVE FAILED: NotImplementedError")
            print(e)
            #stats.unimp += 1
            #stats.solvefailed += 1
        except (SyntaxError, ζ.dsl.DSLError, ζ.ζ3.MalformedError) as e:
            print("-------ζOLVE FAILED")
            print(e)
            #stats.solvefailed += 1
        except Exception as e:
            # Could be a OOM
            print("-------ζOLVE UNCAUGHT EXCEPTION")
            print(e)


if __name__ == '__main__':
    PROMPT_FILE = "prompt.txt"
    ζol = ζolver("Solve THIS!")
    class dummy_makegen:
        new_output = " Yes"
        def append_prompt(*args, **kwargs): pass
        def generate(*args, **kwargs): pass

    ζol.doit(None, None, dummy_makegen, 60)
    print(ζol.scorelog.outputs, ζol.best)
