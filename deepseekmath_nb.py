# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -_cell_guid,-_uuid,-papermill
#     comment_magics: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# credits:
# 
# Directly forked from https://www.kaggle.com/code/anrenk/aimo-llm-usage-clean-code
# which derives from
## https://www.kaggle.com/code/abdurrafae/improved-code-interpretation
# https://www.kaggle.com/code/dnyaneshwalwadkar/submission-with-the-best-nb-new-api
# https://www.kaggle.com/code/utsavsinghal2604/natural-language-and-code-integration
# The core implementation of prompting and executing python is apparently ultimately from
# https://www.kaggle.com/code/olyatsimboy/aimo-zero-shot-sc-mmos-deepseekmath/notebook



# %%
### %%time
import os
# Try reduce wasted memory due to small 11MB allocations consuming 20MB blocks
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  #"max_split_size_mb:256"  #
#PYTORCH_NO_CUDA_MEMORY_CACHING=1

import time
NOTEBOOK_START_TIME = time.time()
import torch

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    PRIVATE = True
else:
    PRIVATE = False

SLOW = True # = PRIVATE
LOGGING = not PRIVATE
VALIDATE = "AMC_12_valid"

if PRIVATE:
    SLOW = True
    VALIDATE = False
OLDCODE = False  # Original code runner
TB = True # Show traceback, OLDCODE=False only
ASK_CONFIDENCE = False  # "Is it proven?"
P100 = (torch.cuda.device_count() == 1)
QUANT = False
USE_PAST_KEY = False
SEED = 314
MODEL_PATH = "/kaggle/input/deepseek-math"
LLEMMA = False   # LLEMMA tokenizer
RELOAD_MODEL = False   # For interactive run-all
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_REPETITIONS = 3 if VALIDATE else (20 if SLOW else 1)   # 6
MAX_SINGLE_GEN_TOKENS = 2048 #1500
MAX_GEN_TOKENS = 2048 if SLOW else 500  #CHANGE
#MAX_TOKENS = 1500 if (P100 and USE_PAST_KEY) else 2048
MAX_TOKENS = 2200

FIRSTPROB = 0  # ignored for PRIVATE

if PRIVATE:
    NPROBS = 50
    TIME_LIMIT = 31000
elif VALIDATE:
    NPROBS = 30 #100
    TIME_LIMIT = 5000
else:
    NPROBS = 1  #10
    TIME_LIMIT = 450
    
LOG_TAG = f"{NPROBS}ofAMC12V_Q{QUANT if QUANT else 'off'}_{MAX_GEN_TOKENS}-{MAX_SINGLE_GEN_TOKENS}tok_{'P100' if P100 else '2xT4'}{'' if USE_PAST_KEY else '_noKV'}{'' if TB else '_noTB'}{'_didprove' if ASK_CONFIDENCE else ''}"
LOG_NAME = time.strftime("%Y%m%d-%H%M-") + LOG_TAG + ".csv"
   

print(f"P100={P100}, DEVICE={DEVICE}, QUANT={QUANT}, SLOW={SLOW}")


# %% [markdown]
# # Install and import libraries, aimo/dummy module

# %% _kg_hide-input=true
%%time
if QUANT and not 'installed_libs' in globals():
    # Need more recent accelerate
    !pip install -U /kaggle/input/accelerate-0-29-3/accelerate-0.29.3-py3-none-any.whl -qq
    !pip install -U /kaggle/input/bitsandbytes-0-43-1/bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl -qq
    installed_libs = True

# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import re
import sys
import subprocess
import math
import random
from collections import defaultdict
from collections import Counter
import torch
import transformers
import accelerate

# %%
if not PRIVATE:
    class train_env():
        def __init__(self, randomize=False):
            if VALIDATE:
                self.df = pd.read_csv(f'/kaggle/input/artofproblemsolving/{VALIDATE}.csv')
            else:
                self.df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/train.csv')
                #self.df.index += 1
                #self.df = self.df.loc[[1,4,6,10]]   # Easiest problems
                #self.df = self.df.reset_index(drop=True)
            
            self.df['ground_truth'] = self.df['answer']
            self.df['answer'] = -1

            if randomize:  # Shuffle
                self.df = self.df.reset_index().sample(frac=1).reset_index(drop=True)

            self.counter = FIRSTPROB
            self.end = min(FIRSTPROB + NPROBS, len(self.df))

        def iter_test(self):
             while self.counter < self.end:
                test = self.df.loc[[self.counter]][['id','problem']]
                sample_submission = self.df.loc[[self.counter]][['id','answer']]

                if VALIDATE:
                    self.prob_id = self.df['prob_name'][self.counter]
                else:
                    self.prob_id = self.df['id'][self.counter]
                print("\n\n\n\n############# env: Prob", self.counter, "of", VALIDATE, "ground_truth =", self.df['ground_truth'][self.counter], "###########")
                print("Prob", self.prob_id)
                yield test, sample_submission
                
        def ground_truth(self):
            return self.df['ground_truth'][self.counter]

        def predict(self, answer):
            self.df.loc[self.counter, ('answer')] = answer['answer'].values[0]
            self.counter += 1

    make_env = train_env
else:
    # Set up the evaluation API
    import aimo

    make_env = aimo.make_env


# %% [markdown]
# # Code processing

# %%
def clean_traceback(output):
    lines = output.strip().split("\n")
    if len(lines) > 20:
        lines = lines[:10] + lines[-10:]
    clean_lines = []
    lineit = iter(lines)
    dots = False
    for i, line in enumerate(lineit):
        # if re.match("^\s*raise "):
        #     line = "    raise"  # Omit
        if line.startswith("Traceback"):
            line = "Traceback"
        # Skip the toplevel function call, since DeepSeek-Math always puts the code in a function
        if 'input.py' in line and '<module>' not in line:  #  File ".../input.py", line...
            line = re.sub('".*input.py"', '"input.py"', line)
        elif re.match('^\s*File "(.*)"', line):
            # Skip this and the next line
            next(lineit)
            dots = True
            continue
        if dots:
            clean_lines.append("...")
            dots = False
        clean_lines.append(line)

    return "\n".join(clean_lines)

def process_code(code):
    
    def repl(match):
        #print("repl run on match:", match.group())
        if "real" not in match.group():
            #print("...adding real=True!")
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    code = "from sympy import *\n" + code
    code = re.sub(r"symbols\([^)]+\)", repl, code)
    # Add a try...except block
    #code = code.replace('\n', '\n    ')
    #code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)

    with open('input.py', 'w') as fout:
        fout.write(code)
    #batcmd = 'timeout 7 ' + sys.executable + ' code.py'
    
    code_status = False
    try:
        process = subprocess.run(sys.executable + ' input.py', shell = True, timeout = 14, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        shell_output = (process.stdout + process.stderr).decode('utf8')
        #shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        print(f"<<<<<###<Result :\n" + shell_output + ">###>>>>>")

        if process.returncode:
            code_status = False
            if TB:
                return_value = clean_traceback(shell_output)
            else:
                return_value = shell_output.strip().split("\n")[-1]  # Last line of exception  ######
            #if "not defined" in return_value:
            #    return_value += '\nTry checking the formatting and imports'
        else:
            return_value = shell_output.strip()
            code_status = True
    except subprocess.TimeoutExpired:
        return_value = "python subprocess timed out. Code too slow."
    #except CalledProcessError as e:
    #    if e.returncode == 124:  # From timeout command
    #        return_value = "python subprocess timed out. Code too slow."
    except Exception as e:
        print('shell_output exception:', e)
        return_value = "python error"  # ??

    return return_value, code_status


# %% [markdown]
# # Text processing

# %%
def naive_parse(text):
    "Naive parsing function to find the last number in a string"
    numbers = re.findall("[0-9]+\.?[0-9]*", text)
    if not numbers:
        return ""
    return numbers[-1]

def process_text_output(result):
    boxed = False
    try:
         
        #result_output = re.findall(r'\\boxed\{(-?\d+[.]?\d*(?:e\d+)?)\}', result)
        result_output = re.findall(r'\\boxed\s*{\s*(\d[^}]*)}', result)

        if len(result_output) == 0:
            result_output = re.findall(r'he answer is:?\s*\$*(\d+)', result)
            if len(result_output) == 0:
                result_output = naive_parse(result)
                print('NAIVE', result_output)
            else:
                result_output = result_output[-1]
                print('ANSWER', result_output)
                boxed = True
        else:
            result_output = result_output[-1]
            # There might be some units, like "\boxed{20\%}"
            if '\\' in result_output and 'frac' not in result_output:
                result_output = result_output.split("\\")[0]
                print('CLEANED BOXED', result_output)
            else:
                print('BOXED', result_output)
            boxed = True

        #print('BOXED FINAL', result_output)
        if len(result_output) == 0:
            result_output = -1
        else:
            result_output = round(float(eval(result_output)))
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    
    return result_output, boxed


# %% [markdown]
# # OLD Code process

# %% _kg_hide-input=true
#####################################

if OLDCODE:

    def return_last_print(output, n):
        lines = output.strip().split('\n')
        if lines:
            return lines[n]
        else:
            return ""

    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')

    def process_code(code, return_shell_output=True):

        code = re.sub(r"symbols\([^)]+\)", repl, code)

        if return_shell_output:
            code = code.replace('\n', '\n    ')
                # Add a try...except block
            code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)

        if not return_shell_output:
            print(code)
        with open('code.py', 'w') as fout:
            fout.write(code)

        batcmd = 'timeout 7 ' + sys.executable + ' code.py'
        try:
            shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
            return_value = return_last_print(shell_output, -1)
            print(shell_output)
            if return_shell_output:
                if return_value=='FAIL':
                    CODE_STATUS = False
                    return_value = return_last_print(shell_output, -2)
                    if "not defined" in return_value:
                        return_value+='\nTry checking the formatting and imports'
                else:
                    CODE_STATUS = True
                return return_value, CODE_STATUS  
            code_output = round(float(eval(return_value))) % 1000
        except Exception as e:
            print(e,'shell_output')
            code_output = -1

        if return_shell_output:
            if code_output==-1:
                CODE_STATUS = False
            else:
                CODE_STATUS = True
            return code_output, CODE_STATUS  
    



# %% [markdown]
# # Util

# %%
def show_gpu_mem():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"CUDA dev {i}: free={free/2**30 :.2f} used={alloc/2**30 :.2f} + reserved={(reserved - alloc)/2**30:.2f} / {total/2**30:.1f} GB")
        
def gpu_mem():
    ret = ""
    for i in range(torch.cuda.device_count()):
        reserved = torch.cuda.memory_reserved(i)
        ret += f"(cuda{i}: {reserved/2**30 :.2f}GB) "
    return ret

def cpu_time() -> str:
    threadt = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
    processt = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
    return f"{threadt:.1f}/{processt:.1f}s CPU"

def show_model_mem(m):
    class MemUse: params, bufs = 0, 0
    counts = defaultdict(MemUse)
    for buf in model.parameters():
        counts[buf.device].params += buf.numel() * buf.itemsize
    for buf in model.buffers():
        counts[buf.device].bufs += buf.numel() * buf.itemsize
    print("Model parameters+buffers:")
    for dev in counts:
        print(f"  {dev}: {counts[dev].params / 2**30 :.2f} + {counts[dev].bufs / 2**30 :.2f} GB")
    mem = model.get_memory_footprint()
    print("  Total:", "%.2f GB" % (mem / 2**30))


# %% [markdown]
# # Load model

# %%
%%time
transformers.set_seed(SEED)

model_kwargs = {}

if QUANT == 4:
    model_kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,  #True would save additional 0.4bits per param
    )
if QUANT == 8:
    model_kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
        load_in_8bit = True,
    )

config = transformers.AutoConfig.from_pretrained(MODEL_PATH)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

config.gradient_checkpointing = True # Enable gradient checkpointing for memory optimization
config.pad_token_id = tokenizer.pad_token_id

LAYERS_GPU0 = 32 if P100 else 18
device_map = [('model.embed_tokens', 0)] + [(f'model.layers.{i}', 0 if i < LAYERS_GPU0 else 1) for i in range(0, 31 + 1)] + [
                 ('model.norm', 1),
                 ('lm_head', 1)]
device_map = {ii:jj for (ii,jj) in device_map}

if P100:
    #model_kwargs['device_map'] = "auto"
    model_kwargs['device_map'] = "sequential"
    #model_kwargs['device_map'] = device_map
else:
    if QUANT:
        # Fits on one device
        model_kwargs['device_map'] = "sequential"
    else:
        model_kwargs['device_map'] = device_map

if not 'model' in globals() or RELOAD_MODEL:
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        trust_remote_code=True,
        config=config,
        **model_kwargs
    )

# Disable memory-efficient sparse tensors for CUDA operations
torch.backends.cuda.enable_mem_efficient_sdp(False)

#print(model)
print()
print("Model", MODEL_PATH)
print("dtype", model.dtype, model.hf_device_map)

show_model_mem(model)
show_gpu_mem()


# %%
class StoppingCriteriaSub(transformers.StoppingCriteria):
    def __init__(self, stops = []):
        super().__init__()
        stoplists = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'][0] for stop_word in stops]
        self.stops = [stop.to("cuda") for stop in stoplists]
        self.ignore_count = 0
        #self.seen = 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            suffix = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop, suffix)):
                if self.ignore_count <= 0:
                    return True
                print("  !ignoring stop!")
                self.ignore_count -= 1
        return False
    
    
def llemma_tok_line(line):
    "Removes the leading newline"
    print(repr(line))
    remove_nl = False
    if not line.startswith("\n"):
        line = "\n" + line
        remove_nl = True
    assert tokenizer("\n", add_special_tokens = False)['input_ids'][-1] == 13
    toks = tokenizer(line, add_special_tokens = False)['input_ids']
    start = toks.index(13) + (1 if remove_nl else 0)
    print(toks, "->",toks[start:])
    return torch.tensor(toks[start:])

class StoppingCriteriaSub(transformers.StoppingCriteria):
    def __init__(self, stops = [], llemma = LLEMMA):
        super().__init__()
        
        self.stop_words = []
        self.stops = []
        for stop_word in stops:
            if llemma:
                toks = llemma_tok_line(stop_word)
            else:
                toks = tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            if stop_word.endswith("DROP"):
                toks = toks[:-1]
                stop_word = stop_word.replace("DROP", "")
            self.stops.append(toks.to(model.device))
            self.stop_words.append(stop_word)
        self.ignore_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for i, stop in enumerate(self.stops):
            suffix = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop, suffix)):
                #print("found", repr(self.stop_words[i]), self.ignore_count)
                if self.ignore_count <= 0:
                    return True
                print("  ignoring stop!")
                self.ignore_count -= 1
        return False
    
BEFORE_DOCSTRING = "():\n   "  # 3 spaces! A 4th space is part of the next token!

# ```->[10897], ````->[4885, 4885], `````->[4885, 10897], ``````->[4885, 4885, 4885]
stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , ")\n\n```" , "```\n", "````"] #, BEFORE_DOCSTRING]
main_stoplist = StoppingCriteriaSub(stop_words)
error_stoplist = StoppingCriteriaSub([" error", " mistake"])

stopping_criteria = transformers.StoppingCriteriaList([main_stoplist])
error_stopping_criteria = transformers.StoppingCriteriaList([main_stoplist, error_stoplist])

#torch.cuda.empty_cache()
#gc.collect()

# %% [markdown]
# # Prompts

# %% _kg_hide-input=false
code = ("""Dear professor, consider this math problem:
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions. Your final answer should be non-negative integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.""",
"Approach:")

code2 = ("""Consider this math problem:
\"{}\"
First, logically analyze the implications of the problem statement. Second, list the general steps of a Sympy-based approach to calculate the answer. Third, write out commented Sympy code to compute the numerical answer and print the result.
You can run and receive results of multiple code blocks to reach the answer in stages. 
Note that intermediate calculations may be real numbers.
Finally, output the final integer answer (not an algebraic expression) within \\boxed{{}}.""",)


cot = ("""Below is a math problem you are to solve:
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final integer answer within \\boxed{{}}.""",)

prompt_options = [code2, code, cot]

# Original prompts

# You can run multiple code blocks to reach the answer in steps.

#similar to 'code' in V9 soln, but several changes
elab0 = ("elab0", """Below is a math problem you are to solve (positive numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. 
You can run multiple code blocks to reach the answer in steps.
Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{{}}.""",
"Approach:")

# Much closer to original V9. Replaces last line
elab0rl = ("elab0rl", """Below is a math problem you are to solve (the answer is a positive integer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step.
Be clear so even an idiot can follow your instructions, and remember, your final answer should be an integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result."""
           "\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
"Approach:")

# With tool prompt
elab0tool = ("elab0tool", """Solve this math problem (the answer is a non-negative integer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step.
Be clear so even an idiot can follow your instructions, and remember, your final answer should be an integer, not an algebraic expression!
Write the entire script covering all the steps and using comments and print the result."""
           "\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.",
"Approach:")

elab1 = ("elab1", """Below is a math problem you are to solve (positive numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. 
Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result.
Don't try the same thing repeatedly if it doesn't work.
Put your final integer answer within \\boxed{{}}.""",
"Approach:\n")

# cot0 == cot in V9
# cot1 replaces 'numerical' with 'integer'
cot1 = ("cot1", """Below is a math problem you are to solve (positive integer answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. """
"""After solving the problem, output the final integer answer within \\boxed{{}}.""",
       )

cot2 = ("cot2", """Below is a math problem you are to solve (non-negative integer answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. """
"""Output the final answer within \\boxed{{}}.""",
       )

cot1rl = ("cot1rl", """Below is a math problem you are to solve (the answer is a positive integer!):
\"{}\"
Analyze this problem and calculate the solution using python."""
"\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
"")

tool0 = ("tool0", """{}
The answer is a non-negative integer.
\nPlease alternate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.""",
"")

tool1 = ("tool1", """"{}"
The answer is a non-negative integer.
Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.""",
"Approach:")



cot2 = ("cot2",  """Below is a math problem you are to solve (positive integer answer!):
\"{}\"

Write an efficient python program to solve it. Write out the whole program and print the result so it will run. If it doesn't work, don't try the same thing repeatedly. Be concise. Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "Our approach")


cot3 = ("shortpkg0", """Here's a problem, with a positive integer answer!
\"{}\"
Analyze step by step and use python/sympy/numpy/scipy/etc to do any calculations or find solutions. After solving the problem, output the final integer answer within \\boxed{{}}.""",
       "")

# analy0 == code2 in V9
# analy1 adds "No docstrings."
analy1 = ("analy1",
    """Consider this math problem:
\"{}\"
First, analyze the implications of the problem statement and restate it more mathematically. Write code to check assumptions, to simplify the problem.
Write out commented Sympy code to compute the numerical answer and print the result. But no docstrings.
Think step by step to come to a solution with programs. After solving the problem, output the final integer answer within \\boxed{{}}.""",
        "")



# Badly behaved, just produces code without thought 
steps0 = ("consise-steps0",  """\"{}\"

Think step by step writing python code to solve this problem. Get to the point. Maths only, no chatting with me. Write out the whole program and print the result.
If it doesn't work and you can't fix it then stop. Put your final answer within \\boxed{{}}. It must be a positive integer.""",
"\n"        )  # Hopefully stops it producing pythogn without ````python

stepver1 = ("con-step-ver1",  """\"{}\"

Think step by step analyzing and writing python code to solve this problem. Get to the point. Maths only, no chatting with me. Write out the whole program and print the result. No docstrings.
Verify your result and stop if it's wrong. Put your final answer within \\boxed{{}}. It must be a positive integer.""",
""        )

verifyXX = ("verify0", """Below is a math problem you are to solve (positive numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. 
Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result.
Verify your answer is correct, with some test code if you can. If it is correct put your final numerical answer within \\boxed{{}}. Otherwise
""",
"Approach:\n")


prompt_options = [elab0, analy1, steps0, stepver1] #, code, code2, cot, steps]  # cot2
# Similar to V9 but missing cot
prompt_options = [elab0, analy1, steps0]

#prompt_options = [elab0rl, cot1rl]  # USed for a number of submissions
prompt_options = [steps0, tool0, tool1]  # Scored 22
#prompt_options = [tool0, tool1, elab0tool]

# %% [markdown]
# # Generation and main loop

# %%

class LLMGenerator:
    def __init__(self):
        self.tokens = tokenizer("")['input_ids']
        self.prompt = tokenizer.decode(self.tokens)  # Will be equal to tokenizer.bos_token
        self.num_gen_tokens = 0  # Total, not including prompt and outputs
        self.past_key_values = None
        self.hit_limit = False
        self.need_update = True
        self.stop_on_error = True
        self.bad_words = None

    def update_inputs(self):
        if True:
            # split_special_tokens = False is the default, so <｜end▁of▁sentence｜> will convert
            self.model_inputs = tokenizer(self.prompt, add_special_tokens = False, return_tensors='pt').to(model.device)
        else:
            self.model_inputs = tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        self.tokens = self.model_inputs['input_ids'][0]
        #self.decode_tokens()  # Puts the 
        
    def decode_tokens(self):
        self.prompt = tokenizer.decode(self.tokens, skip_special_tokens = False)
        self.need_update = False
        
    def user_prompt(self, text, assist_prefix = ""):
        """Add `text` to the prompt as a User message, followed by Assistant:, and optionally start of assistant response.
        Call this before the first .generate() or at any time to add a User direction."""
        old_input = len(self.tokens)
        ol = len(self.prompt)
        if True:
            # DeepSeek specific template
            if old_input > 1 and not self.prompt.endswith(tokenizer.eos_token):
                self.prompt += tokenizer.eos_token
            if True:
                self.prompt += ("User: " + text.strip() + "\n\nAssistant: " + assist_prefix.strip(' ')).strip(' ')
            else:
                self.prompt += ("User: " + text.strip() + "\n\n" + assist_prefix.strip(' ')).strip(' ')
        else:
            self.messages += [{"role": "user", "content": text}]
            if assist_prefix:
                self.messages.append({"role": "assistant", "content": assist_prefix})
        self.update_inputs()
        print(f"<<<<<PROMPT {len(self.tokens) - old_input} tokens\n" + self.prompt[ol:] + ">>>>>")

    def append_prompt(self, text):
        "Append to prompt, without changing the role."
        text = text.rstrip(' ')
        self.prompt += text
        old_input = len(self.tokens)
        self.update_inputs()
        print(f"<<<<<APPEND {len(self.tokens) - old_input} tokens\n" + text + ">>>>>")
        
    def replace_tokens(self, new_tokens):
        self.tokens = new_tokens
        self.past_key_values = None
        self.decode_tokens()
        
    def replace_prompt(self, new_prompt):
        self.prompt = new_prompt
        self.past_key_values = None
        self.update_inputs()
        
    def check_limit(self):
        max_toks = min(MAX_SINGLE_GEN_TOKENS, MAX_TOKENS - len(self.tokens), MAX_GEN_TOKENS - self.num_gen_tokens)
        return max_toks

    def generate(self, temperature = 0.9, top_p = 1.0, limit = 9999):
        startt = time.time()
        max_toks = self.check_limit()
        if max_toks < 3:
            print("SKIP GENERATE DUE TO LIMIT")
            return
       
        #rint("INPUT;", self.model_inputs['input_ids'].shape)
        #f self.past_key_values:
        #   print("PAST: len ", len(self.past_key_values[0][0]))
        
        #input_prompt = tokenizer.decode( self.model_inputs['input_ids'][0], skip_special_tokens = False)
        #print("$$<<<$$ ENTIRE PROMPT\n" + input_prompt + "\n$$>>>$$")
        
        #if self.need_update:
        #    self.update_inputs()  # Regenerate model_input
        if self.stop_on_error:
            stopper = error_stopping_criteria
        else:
            stopper = stopping_criteria
        input_len = len(self.tokens)
        generation_output = model.generate(**self.model_inputs, 
                                           max_new_tokens = min(limit, max_toks),
                                           use_cache = USE_PAST_KEY,
                                           return_dict_in_generate = USE_PAST_KEY,
                                           past_key_values = self.past_key_values,
                                           do_sample = True,
                                           temperature = temperature,
                                           top_p = top_p,
                                           bad_words_ids = self.bad_words,
                                           pad_token_id = tokenizer.eos_token_id,
                                           num_return_sequences = 1,
                                           stopping_criteria = stopper)

        if USE_PAST_KEY:
            sequences = generation_output.sequences
            self.past_key_values = generation_output.past_key_values
            
        else:
            sequences = generation_output
        #print("out len toks = ", len(self.tokens))
        self.tokens = sequences[0]
        decoded_output = tokenizer.decode(self.tokens, skip_special_tokens = False) #True)
        self.new_output = decoded_output[len(self.prompt):]
        self.prompt = decoded_output
        self.need_update = True
        self.model_inputs = {'input_ids': sequences}
        #print("out prompt len = ", len(self.prompt))
        #self.new_output = tokenizer.decode(gen.tokens[input_len:], skip_special_tokens=True)

        runt = time.time() - startt

        new_toks = len(self.tokens) - input_len
        self.num_gen_tokens += new_toks
        if new_toks >= MAX_SINGLE_GEN_TOKENS:
            print("HIT MAX_SINGLE_GEN_TOKENS")
            self.hit_limit = True
        if self.num_gen_tokens >= MAX_GEN_TOKENS:
            print("HIT MAX_GEN_TOKENS")
            self.hit_limit = True   # TODO: instead append  "...\nPutting that into code:\n```python"
        if len(self.tokens) >= MAX_TOKENS:
            print("HIT MAX_TOKENS")
            self.hit_limit = True

        print(f"<<<<<GEN {new_toks} tokens ({len(self.tokens)} total) in {runt :.1f}s ({new_toks/runt :.1f} tok/s) ({cpu_time()}) {gpu_mem()}\n"
              + self.new_output
              + ">>>>>")

    def endswith(self, text):
        return self.prompt.endswith(text)
    
    def set_bad_words(self, words: list):
        self.bad_words = tokenizer(words, add_special_tokens = False)['input_ids']

if LOGGING:
    logdf = pd.DataFrame(columns = ['problem_id', 'prompt', 'score', 'answer', 'result_info', 'gen_tokens', 'code_blocks', 'code_errors', 'time', 'bad'])
    

def predict(probi, problem):

    temperature = 0.75
    top_p = 0.95
    temperature_coding = 0.75
    top_p_coding = 0.95

    firstprompt = random.randint(0, 9999)
    score = 0
    best = 0
    outputs = []  # List of (answer, score, info) tuples
    answer_scores = defaultdict(int)  # answer -> total_score

    time_left = TIME_LIMIT - (time.time() - NOTEBOOK_START_TIME)
    time_for_item = time_left / max(1, NPROBS - probi)
    item_time_start = time.time()
    for jj in range(N_REPETITIONS): # tqdm(range(N_REPETITIONS)):
        it_start_time = time.time()
        time_spent = time.time() - NOTEBOOK_START_TIME
        spent_this_prob = (time.time() - item_time_start)
        print(f"\n\n----QUESTION {probi} - rep.{jj} - time_spent : {time_spent:.0f}/{TIME_LIMIT}, on this prob: {spent_this_prob:.1f}/{time_for_item:.0f} secs")
        if time_spent > TIME_LIMIT or spent_this_prob > time_for_item:
            break
        
        for _ in range(5):
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)
        
        if LOGGING:
            global logdf
            logrow = pd.DataFrame(columns = logdf.columns)
            logrow.loc[len(logdf)] = 0
            logrow.problem_id = env.prob_id
        
        last_code_error = None
        code_status = True
        code_error_count = 0
        code_blocks = 0
        #error_word_count = 0
        error_stoplist.ignore_count = 2  # Stop on the third mention of "error" or "mistake"
        prev_code_len = 0
        bad = False
        result_output = -1
        result_info = "NA"
        code_output = "-1"
        last_code_result = -1
        cumulative_code = ""
        
        try:
            gen = LLMGenerator()
            gen.set_bad_words([' """'])

            promptname, prompt, assist = prompt_options[(firstprompt+jj) % len(prompt_options)]  #random.choice(prompt_options)
            print("prompt", promptname)
            if LOGGING:
                logrow.prompt = promptname
            gen.user_prompt(prompt.format(problem), assist)
            gen.generate(temperature, top_p)
            
            currently_text = True

            while not gen.hit_limit and gen.check_limit() > 3:
                
                if re.search("Assistant: (from|import) ", gen.prompt):   #gen.new_output.startswith(" from "):
                    # ``` is missing
                    print("FIX INITIAL MISSING ```")
                    gen.replace_prompt(gen.prompt.replace("Assistant: ", "Assistant: ```python\n"))
                    currently_text = False
                
                # Process stopwords
                
                if gen.tokens[-1] == tokenizer.eos_token_id:
                    if not currently_text:
                        print("EOS TOKEN ENDS CODE!")
                        gen.replace_tokens(gen.tokens[:-1])
                        gen.append_prompt("\n```\n")
                    else:
                        print("GOT EOS")
                        result_info = "eos"
                        break
                elif gen.endswith("mistake") or (gen.stop_on_error and gen.endswith("error")):  # mistake also in stop_words
                    # This only happens after error_stoplist.ignore_count repetitions
                    print("MISTAKE")
                    bad = True
                    break
                    
                elif not any(gen.endswith(stop_word) for stop_word in main_stoplist.stop_words):
                    # No continuation, and didn't hit limit, so must have ended due to eos.
                    result_info = "no continue"
                    print("nocont", repr(gen.prompt[-10:]))
                    break


                    
                if gen.endswith(BEFORE_DOCSTRING):
                    # Oh you think you're going to copy the problem into the docstring, do you? No!!!
                    gen.append_prompt(" #")
                    temperature_inner = temperature_coding
                    top_p_inner = top_p_coding

                # Model sometimes outputs a line of ```` often not ending in 'python'
                elif gen.endswith("```python") or (currently_text and gen.endswith("````") ):
                    # Starting code
                    temperature_inner = temperature_coding
                    top_p_inner = top_p_coding
                    code_status = True
                    currently_text = False
                    if gen.endswith("````"):
                        gen.append_prompt("python\n")
                else:
                    # Just finished producing code
                    temperature_inner = temperature
                    top_p_inner = top_p
                    currently_text = True

                    code_status = False
                    try:
                        if gen.endswith("``````output"):
                            print("(((Weird ``````output)))")
                            code_text = gen.prompt.split('```python')[-1].split("``````")[0]
                        else:
                            # The code block may end in ```` (or maybe more, but at which point stop should be triggered), that's OK
                            # Also it might end in eos_token if the model didn't start its code with ``` at all
                            code_text = gen.prompt.split('```python')[-1].split("```")[0]

                        if OLDCODE:
                            cumulative_code+=code_text
                            code_output, code_status = process_code(cumulative_code, return_shell_output=True)
                            print('CODE RESULTS', code_output)

                            if last_code_error==code_output:
                                code_error_count+=1
                            else:
                                last_code_error=code_output
                                code_error_count = 0

                            if not code_status:
                                cumulative_code = cumulative_code[:-len(code_text)]

                                if code_error_count>=1:
                                    print("REPEATED ERRORS")
                                    # bad = True ????????
                                    break
                                    
                            try:
                                # code_output is -1 or a string
                                last_code_result = int(code_output)
                            except:
                                pass
                        else:
                            
                            all_code = cumulative_code + code_text
                            code_output, code_status = process_code(all_code)

                            #print('<<<<<CODE RESULTS\n' + code_output + ">>>>>")

                            code_blocks += 1
                            if LOGGING:
                                if not code_status:
                                    logrow.code_errors += 1

                            if code_status == True:
                                code_error_count = 0
                                cumulative_code += code_text
                                new_len = len(code_output)
                                code_output = code_output[prev_code_len:].strip()
                                prev_code_len = new_len
                                try:
                                    last_code_result = round(float(eval(code_output.strip().split("\n")[-1])))
                                except:
                                    pass
                            else:
                                # the last line is the exception
                                except_line = code_output.strip().split("\n")[-1]
                                if except_line == last_code_error:
                                    code_error_count += 1
                                else:
                                    code_error_count = 1
                                last_code_error = except_line
                                if code_error_count >= 2:
                                    bad = True
                                    print("REPEATED ERROR")
                                    break

                    except Exception as e:
                        print(e)
                        print('ERROR PARSING CODE')
                        code_output = ""

                    if OLDCODE and (code_output == "" or code_output == -1):
                        # Add nothing to prompt
                        out = ""
                        cumulative_code = ""
                    #if gen.endswith(")\n```"):
                    elif gen.endswith("```") or gen.endswith("```\n"):
                        out = '```output\n' + str(code_output) + '\n```\n'
                    elif gen.endswith("```output") or gen.endswith("```output\n"):
                        out = str(code_output) + '\n```\n'
                    else:
                        print("(((doesn't end with ``` or ```output)))")
                        if OLDCODE:
                            out = '\n' + str(code_output) + '\n```\n'
                        else:
                            out = '>>> ' + str(code_output) + '\n\n'
                    if not gen.endswith("\n"):
                        out = "\n" + out
                    gen.append_prompt(out)

                    # if code_status:
                    #     #if gen.endswith(")\n```"):
                    #     if gen.endswith("\n```"):
                    #         gen.append_prompt('```output\n' + code_output + '\n```\n')
                    #     else:
                    #         #gen.append_prompt('\n' + code_output + '\n```\n')
                    #         gen.append_prompt('\n>>> ' + code_output + '\n\n')
                    # else:
                    #     pass #cumulative_code = ""

                if code_status == False:
                    error_stoplist.ignore_count += 1
                    #gen.stop_on_error = False   # Don't stop on "error", the LLM may say "to fix the error"
                else:
                    gen.stop_on_error = True
                gen.generate(temperature_inner, top_p_inner)

            boxed = False
            penalty = 0
            confident = ""
            if gen.hit_limit:
                print("HIT LIMIT")
                # In the middle of text or code, the last number is almost certainly not the answer
                result_output = -1
                # Not necessarily bad, maybe can salvage answer from code_output
            elif not bad:
                result_output, boxed = process_text_output(gen.new_output)
                if result_output != -1 and last_code_result != -1 and result_output != last_code_result:
                    print("HMM... TEXT/CODE MISMATCH")
                    penalty = 0.1
                    # BAd idea since the agent will almost certainly agree
                    #if gen.check_limit() > 45:
                    #    gen.user_prompt(f"Is the answer actually {last_code_result}? If you know it put it in \\boxed{{}}")
                    #    gen.generate(0.2, top_p, 20)
                    #    result_output, boxed = process_text_output(gen.prompt)
                    #    if result_output == last_code_result:
                    #        penalty = 0
                    
                if not boxed and gen.check_limit() > 23:
                    # Trying again
                    print("FORCING BOXED")
                    gen.user_prompt("If you know the answer put it in \\boxed{}")
                    gen.generate(0.2, top_p)
                    result_output, boxed = process_text_output(gen.prompt)
                if ASK_CONFIDENCE and penalty == 0 and boxed and gen.check_limit() > 23:
                    gen.user_prompt(ASK_CONFIDENCE)
                    gen.generate(0.2, top_p, 3)
                    confident = gen.new_output.lower()
            
            if bad:
                pass
            elif result_output <= -1:
                # Fallback
                try:
                    result_output = last_code_result
                    print("code_output fallback got:", result_output)
                    result_info = "code_output"
                    score = 0.4
                    if bad:
                        score = 0.2
                except Exception as e:
                    print(e, 'code_output fallback parse failed')
            elif not bad:
                if boxed:
                    score = 1
                else:
                    score = 0.7
                if 'yes' in confident:
                    score += 0.05
                if 'no' in confident:
                    score -= 0.1
                    
            #if gen.num_gen_tokens > 1200:
            #    score -= 0.2
            #elif gen.num_gen_tokens > 950:
            #    score -= 0.1
            if 1 <= code_blocks <= 3:
                score += 0.15
            score -= penalty
                    
        except torch.cuda.OutOfMemoryError as e:
        #except Exception as e:
            print("predict() EXCEPTION")
            print(e)
            #result_output = -1
            bad = 2
        
        if LOGGING:
            # ['problem_id', 'prompt', 'score', 'answer', 'result_info', 'gen_tokens', 'code_errors', 'time'])
            logrow.result_info = result_info
            logrow.gen_tokens = gen.num_gen_tokens
            logrow.code_blocks = code_blocks
            logrow.score = score
            logrow.answer = result_output
            logrow.time = time.time() - it_start_time
            logrow.bad = bad
            logdf = pd.concat([logdf, logrow])
            logdf.to_csv(LOG_NAME)

        if result_output > -1:  # and not bad
            print(f"RESULT = {result_output} SCORE = {score}")
            result_output = result_output % 1000
            outputs.append((result_output, score, result_info))
            answer_scores[result_output] += max(0, score)
            
            

        if len(outputs) > 0:
            answers = [(score,ans) for (ans,score) in answer_scores.items()]
            answers.sort(reverse = True)
            print("SCORES,ANSWERS:", answers)
            best_score, best = answers[0]
            #if best_score >= 3 and best_score >= 1 + (jj+1)/2:
            if best_score > 4 and not VALIDATE:
                print("ANSWER FOUND!")
                break   # ####FIXME

    print("\nAll outputs:", outputs)
    return best

# %%
env = make_env()
iter_test = env.iter_test()

if not PRIVATE:
    NOTEBOOK_START_TIME = time.time()
for probi, (test, sample_submission) in enumerate(iter_test):
    sample_submission['answer'] = predict(probi, test['problem'].values[0])
    #print(f"Making prediction for ""{test[:100]}"": {sample_submission}")
    env.predict(sample_submission)

# %%
if not PRIVATE:
    if LOGGING:
        with open("prompts.txt", "w") as fo:
            fo.write(repr(prompt_options))
    print(env.df)
    score = (env.df.ground_truth == env.df.answer).sum()
    print(f'{score} matches in {len(env.df)} examples')
