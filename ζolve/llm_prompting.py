
INDIR = "/kaggle/input/olve-prompts/"

def build_prompt(filename = INDIR + "prompt.txt", sections = ['seqs', 'graphs', 'complex', 'ntheory'], startcode = '```', endcode = '```', maxlength = 10000):

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

    prompt = selected
    prompt = prompt.replace('STARTCODE', startcode).replace('ENDCODE', endcode).replace("STARTANSWER", startanswer)

    return prompt

print(build_prompt("prompt.txt"))

def topic_query(gen, problem):
    "gen is a fresh LLMGenerator"

    prompt = r"""# User instruction
Categorise maths problems by topic:

# Problem
"What is the sum of all values of $y$ for which the complex function $\frac{y+6}{y^2-5y+4}$ is undefined?"

# Topics
Graphs of functions: Yes
Combinatorics: No
Probability: No
Number theory: No
Complex numbers: Yes
Linear algebra: No
Recurrence relations: No
Sequences: No
Geometry: No

# Problem
"Let $S$ be a region in the plane with area 10.  When we apply the matrix \[\begin{pmatrix} 2 & 1 \\ 7 & -3 \end{pmatrix}\]to $S,$ we obtain the region $S'.$  Find the area of $S'.$"

# Topics
Graphs of functions: No
Combinatorics: No
Probability: No
Number theory: No
Complex numbers: No
Linear algebra: Yes
Recurrence relations: No
Sequences: No
Geometry: Yes

# Problem
"The Smith family has 4 sons and 3 daughters. In how many ways can they be seated in a row of 7 chairs such that no boys are next to each other?"

# Topics
Graphs of functions: No
Combinatorics: Yes
Probability: No
Number theory: No
Complex numbers: No
Linear algebra: No
Recurrence relations: No
Sequences: Yes
Geometry: No

# Problem
"Find the smallest four-digit palindrome which is the cube of a prime."

# Topics
Graphs of functions: No
Combinatorics: No
Probability: No
Number theory: Yes
Complex numbers: No
Linear algebra: No
Recurrence relations: No
Sequences: Yes
Geometry: No

# Problem
"{problem}"

# Topics"""


    topics = """graphs|Graphs of functions:
comb|Combinatorics:
prop|Probability:
ntheory|Number theory:
complex|Complex numbers:
linalg|Linear algebra:
relations|Recurrence relations:
seqs|Sequences:
geometry|Geometry:""".split("\n")

    gen.append_prompt(prompt.format(problem))

    sections = []
    for topic in topics:
        code, name = topic.split("|")
        gen.append_prompt("\n" + topic)
        gen.generate(0.1, top_p = 0.1, limit = 1)
        if gen.new_output == " Yes":
            sections.append(code)
        if gen.new_output not in (" Yes", " No"):
            print("WARNING: UNRECOGNISED TOPIC ANSWER:", gen.new_output)

    print(sections)
    return sections

if False:
    # Remove comments
    lines = []
    for line in transprompt.split("\n"):
        if line.startswith('# '): continue
        lines.append(line)
    #transprompt = ('\n'.join(lines))
