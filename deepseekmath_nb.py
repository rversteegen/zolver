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

# %% jupyter={"outputs_hidden": false}
# credits:
# https://www.kaggle.com/code/abdurrafae/improved-code-interpretation
# https://www.kaggle.com/code/dnyaneshwalwadkar/submission-with-the-best-nb-new-api
# https://www.kaggle.com/code/utsavsinghal2604/natural-language-and-code-integration
# Forked from https://www.kaggle.com/code/anrenk/aimo-llm-usage-clean-code

# %% jupyter={"outputs_hidden": false}
import time

NOTEBOOK_START_TIME = time.time()
print(NOTEBOOK_START_TIME)

# %% [markdown]
# # Libraries installation

# %% jupyter={"outputs_hidden": false}
%%time
try:
    import accelerate
except:
    !pip install -U /kaggle/input/accelerate-0-29-3/accelerate-0.29.3-py3-none-any.whl -qq
    !pip install -U /kaggle/input/bitsandbytes-0-43-1/bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl -qq

# %% jupyter={"outputs_hidden": false}
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

# %% jupyter={"outputs_hidden": false}
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    PRIVATE = True
else:
    PRIVATE = False

DEBUG = False
P100 = (torch.cuda.device_count() == 1)
QUANT = False
USE_PAST_KEY = True
SEED = 314
MODEL_PATH = "/kaggle/input/deepseek-math"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_REPETITIONS = 20 if PRIVATE else 2
MAX_GEN_TOKENS = 1400 if PRIVATE else 300  #CHANGE
MAX_SINGLE_GEN_TOKENS = 850 #1500
MAX_TOKENS = 2048
    
if PRIVATE:
    NPROBS = 50
    TIME_LIMIT = 31000
else:
    NPROBS = 4
    TIME_LIMIT = 500
    
print(P100, DEVICE)

# %% jupyter={"outputs_hidden": false}
if not PRIVATE:
    class train_env():
        def __init__(self, randomize=False):
            self.df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/train.csv')
            self.df['ground_truth'] = self.df['answer']
            self.df['answer'] = -1

            self.df.index += 1
            self.df = self.df.loc[[1,4,6,10]]
            self.df = self.df.reset_index(drop=True)

            if randomize:
                self.df = self.df.reset_index().sample(frac=1).reset_index(drop=True)

            self.predict_called = True
            self.counter = 0
            self.len = len(self.df)

        def iter_test(self):
             while self.counter < self.len:
                if self.predict_called:
                    self.predict_called = False
                    test = self.df.loc[[self.counter]][['id','problem']]
                    sample_submission = self.df.loc[[self.counter]][['id','answer']]
                    print("env: ground_truth =", self.df['ground_truth'][self.counter])
                    yield test, sample_submission
                else:
                    print("You must call `predict()` successfully before you can continue with `iter_test()`")
                    return None  # Prevent loop

        def predict(self, answer):
            self.df.loc[self.counter, ('answer')] = answer['answer'].values[0]
            self.predict_called = True
            self.counter+=1

    make_env = train_env
    #env = train_env(randomize=True)
else:
    # Set up the evaluation API
    import aimo

    make_env = aimo.make_env
    #env = aimo.make_env()


# %% [markdown]
# # Important Custom Functions

# %% jupyter={"outputs_hidden": false}
def output_line(output, n):
    lines = output.strip().split('\n')
    try:
        return lines[n]
    except IndexError:
        return ""

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
        if 'code.py' in line:  #  File ".../code.py", line...
            line = re.sub('".*code.py"', '"input.py"', line)
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
    code = re.sub(r"symbols\([^)]+\)", repl, code)
    # Add a try...except block
    #code = code.replace('\n', '\n    ')
    #code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)

    with open('code.py', 'w') as fout:
        fout.write(code)
    #batcmd = 'timeout 7 ' + sys.executable + ' code.py'
    
    code_status = False
    try:
        process = subprocess.run(sys.executable + ' code.py', shell = True, timeout = 2, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        shell_output = (process.stdout + process.stderr).decode('utf8')
        #shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        print(f"<<<<<###<Result :\n" + shell_output + ">###>>>>>")

        if process.returncode:
        #if output_line(shell_output, -1) == 'FAIL':
            code_status = False
            return_value = clean_traceback(shell_output)
            #return_value = output_line(shell_output, -2)  # Last line of exception
            if "not defined" in return_value:
                return_value += '\nTry checking the formatting and imports'
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
        result_output = re.findall(r'\boxed{(.*)}', result)

        if len(result_output) == 0:
            result_output = naive_parse(result)
            print('NAIVE', result_output)
            #result_output = ""
            #return -1
        else:
            result_output = result_output[-1]
            print('BOXED', result_output)
            boxed = True

        #print('BOXED FINAL', result_output)
        if len(result_output) == 0:
            result_output = -1
        else:
            result_output = round(float(eval(result_output))) % 1000
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    
    return result_output, boxed

def cpu_time() -> str:
    threadt = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
    processt = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
    return f"{threadt:.1f}/{processt:.1f}s CPU"


# %% [markdown]
# # Start of code

# %% jupyter={"outputs_hidden": false}
transformers.set_seed(SEED)

model_kwargs = {}

if QUANT:
    model_kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

config = transformers.AutoConfig.from_pretrained(MODEL_PATH)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

config.pad_token_id = tokenizer.pad_token_id

LAYERS_GPU0 = 32 if P100 else 18
device_map = [('model.embed_tokens', 0)] + [(f'model.layers.{i}', 0 if i < LAYERS_GPU0 else 1) for i in range(0, 31 + 1)] + [
                 ('model.norm', 1),
                 ('lm_head', 1)]
device_map = {ii:jj for (ii,jj) in device_map}

if P100:
    model_kwargs['device_map'] = "auto"
    model_kwargs['device_map'] = "sequential"
    #model_kwargs['device_map'] = device_map
else:
    if QUANT:
        # Fits on one device
        model_kwargs['device_map'] = "sequential"
    else:
        model_kwargs['device_map'] = device_map

if QUANT:
    quantization_config=quantization_config

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

print(model)
print()
print(model.dtype, model.hf_device_map)

mem = model.get_memory_footprint()
print("mem usage:", "%.2fGB" % (mem / 2**30))

def show_mem():
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
        
show_mem()


# %% jupyter={"outputs_hidden": false}
class StoppingCriteriaSub(transformers.StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            suffix = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop, suffix)):
                return True
        return False

stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , "```\n", "\n```\n", "``````output"] #,  
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = transformers.StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


# %% jupyter={"outputs_hidden": false}
torch.cuda.empty_cache()
gc.collect()

# %% jupyter={"outputs_hidden": false}
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
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{{}}.""",)

prompt_options = [code2, code, cot]

# Original prompts

# You can run multiple code blocks to reach the answer in steps.
code = ("""Below is a math problem you are to solve (positive numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. 
Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result.
Don't try the same thing repeatedly if it doesn't work.
Put your final numerical answer within \\boxed{{}}.""",
"Approach:\n")


cot = ("""Below is a math problem you are to solve (positive numerical answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{{}}.""",)

cot2 = ("""Here's a problem, with a positive integer answer!
\"{}\"
Analyze step by step and use python/sympy/numpy/scipy/etc to do any calculations or find solutions. After solving the problem, output the final numerical answer within \\boxed{{}}.""",)



code2 = ("""Consider this math problem:
\"{}\"
First, analyze the implications of the problem statement and restate it more mathematically. Write code to check assumptions, to simplify the problem.
Write out commented Sympy code to compute the numerical answer and print the result.
Think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{{}}.""",)


cot = ("""Below is a math problem you are to solve (positive numerical answer!):
\"{}\"

Write a python program to solve it. If it doesn't work, don't try the same thing repeatedly. Be concise. Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "Our approach")

prompt_options = [code, cot2] #, cot2]


# %% jupyter={"outputs_hidden": false}

class LLMGenerator:
    def __init__(self):
        self.prompt = ""
        self.num_gen_tokens = 0  # Not including prompt and outputs
        self.past_key_values = None
        self.hit_limit = False
        self.need_update = True

    def update_inputs(self):
        if True:
            self.model_inputs = tokenizer(self.prompt, return_tensors='pt').to(model.device)
        else:
            self.model_inputs = tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        self.tokens = self.model_inputs['input_ids'][0]
        self.input_len = len(self.tokens)
        self.decoded_output = tokenizer.decode(self.tokens, skip_special_tokens = False)
        self.prompt = self.decoded_output
        self.need_update = False
        
    def initial_prompt(self, text, assist_prefix = ""):
        if True:
            self.prompt = "User: " + text + "\n\nAssistant: " + assist_prefix
        else:
            self.messages = [{"role": "user", "content": text}]
            if assist_prefix:
                self.messages.append({"role": "assistant", "content": assist_prefix})
        self.update_inputs()
        print(f"<<<<<PROMPT {self.input_len} tokens\n" + self.decoded_output + ">>>>>")

    def append_prompt(self, text):
        self.prompt += text
        old_input = self.input_len
        self.update_inputs()
        print(f"<<<<<APPEND {self.input_len - old_input} tokens\n" + text + ">>>>>")

    def generate(self, temperature = 0.9, top_p = 1.0):
        startt = time.time()

        max_toks = min(MAX_SINGLE_GEN_TOKENS, MAX_TOKENS - self.input_len, MAX_GEN_TOKENS - self.num_gen_tokens)
        #rint("INPUT;", self.model_inputs['input_ids'].shape)
        #f self.past_key_values:
        #   print("PAST: len ", len(self.past_key_values[0][0]))
        
        #input_prompt = tokenizer.decode( self.model_inputs['input_ids'][0], skip_special_tokens = False)
        #print("$$<<<$$ ENTIRE PROMPT\n" + input_prompt + "\n$$>>>$$")
        
        if self.need_update:
            self.update_inputs()
        
        generation_output = model.generate(**self.model_inputs, 
                                           max_new_tokens = max_toks,
                                           use_cache = True,
                                           return_dict_in_generate = USE_PAST_KEY,
                                           past_key_values = self.past_key_values,
                                           do_sample = True,
                                           temperature = temperature,
                                           top_p = top_p,
                                           pad_token_id = tokenizer.eos_token_id,
                                           num_return_sequences = 1,
                                           stopping_criteria = stopping_criteria)

        if USE_PAST_KEY:
            self.tokens = generation_output.sequences[0]
            self.past_key_values = generation_output.past_key_values
            
        else:
            self.tokens = generation_output[0]
        #print("out len toks = ", len(self.tokens))
        self.decoded_output = tokenizer.decode(self.tokens, skip_special_tokens = False) #True)
        self.new_output = self.decoded_output[len(self.prompt):]
        self.prompt = self.decoded_output
        self.need_update = True
        #print("out prompt len = ", len(self.prompt))
        #self.new_output = tokenizer.decode(gen.tokens[self.input_len:], skip_special_tokens=True)

        runt = time.time() - startt

        new_toks = len(self.tokens) - self.input_len
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
        return self.decoded_output.endswith(text)


def predict(probi, problem):

    temperature = 0.96
    top_p = 0.95
    temperature_coding = 0.9
    top_p_coding = 1.0

    score = 0
    best = 0
    outputs = []  # List of (answer, score, info) tuples
    answer_scores = defaultdict(int)  # answer -> total_score

    time_left = TIME_LIMIT - (time.time() - NOTEBOOK_START_TIME)
    time_for_item = time_left / max(1, NPROBS - probi)
    item_time_start = time.time()
    for jj in range(N_REPETITIONS): # tqdm(range(N_REPETITIONS)):
        time_spent = time.time() - NOTEBOOK_START_TIME
        spent_this_prob = (time.time() - item_time_start)
        print(f"\n\n----QUESTION {probi} - rep.{jj} - time_spent : {time_spent:.0f}/{TIME_LIMIT}, on this prob: {spent_this_prob:.1f}/{time_for_item:.0f} secs")
        if time_spent > TIME_LIMIT or spent_this_prob > time_for_item:
            break
        
        for _ in range(5):
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)
        
        last_code_error = None
        code_error_count = 0
        prev_code_len = 0
        bad = False
        result_info = "NA"
        code_output = "-1"
        last_code_result = -1
        cumulative_code = ""

        try:
            gen = LLMGenerator()

            prompt = list( prompt_options[jj % len(prompt_options)] )
            prompt[0] = prompt[0].format(problem)
            gen.initial_prompt(*prompt)
            gen.generate(temperature, top_p)

            while not gen.hit_limit:
                if gen.tokens[-1] == tokenizer.eos_token_id:
                    result_info = "eos"
                    break
                if not any(gen.endswith(stop_word) for stop_word in stop_words):
                    # No continuation, and didn't hit limit, so must have ended due to eos.
                    result_info = "no continue"
                    break

                # TODO: have seen ``` instead of ```python
                if gen.endswith("```python"):
                    temperature_inner = temperature_coding
                    top_p_inner = top_p_coding
                else:
                    temperature_inner = temperature
                    top_p_inner = top_p

                    code_status = False
                    try:
                        if gen.endswith("``````output"):
                            print("(((Weird ``````output)))")
                            code_text = gen.decoded_output.split('```python')[-1].split("``````")[0]
                        else:
                            code_text = gen.decoded_output.split('```python')[-1].split("```")[0]

                        all_code = cumulative_code + code_text
                        code_output, code_status = process_code(all_code)
                        #print('<<<<<CODE RESULTS\n' + code_output + ">>>>>")

                        if code_status == True:
                            code_error_count = 0
                            cumulative_code += code_text
                            new_len = len(code_output)
                            code_output = code_output[prev_code_len:]
                            prev_code_len = new_len
                            try:

                                last_code_result = round(float(eval(code_output.strip().split("\n")[-1]))) % 1000
                            except:
                                pass
                        else:
                            # code_output is the exception line
                            if code_output == last_code_error:
                                code_error_count += 1
                            else:
                                code_error_count = 1
                            last_code_error = code_output
                            if code_error_count >= 2:
                                bad = True
                                print("REPEATED ERROR")
                                break

                    except Exception as e:
                        print(e)
                        print('ERROR PARSING CODE')
                        code_output = ""

                    #if gen.endswith(")\n```"):
                    if gen.endswith("```") or gen.endswith("```\n"):
                        nl = "\n"
                        if gen.endswith("\n"):
                            nl = ""
                        gen.append_prompt(nl + '```output\n' + code_output + '\n```\n')
                    else:
                        print("(((doesn't end with \\n```)))")
                        #gen.append_prompt('\n' + code_output + '\n```\n')
                        gen.append_prompt('\n>>> ' + code_output + '\n\n')

                    # if code_status:
                    #     #if gen.endswith(")\n```"):
                    #     if gen.endswith("\n```"):
                    #         gen.append_prompt('```output\n' + code_output + '\n```\n')
                    #     else:
                    #         #gen.append_prompt('\n' + code_output + '\n```\n')
                    #         gen.append_prompt('\n>>> ' + code_output + '\n\n')
                    # else:
                    #     pass #cumulative_code = ""

                gen.generate(temperature_inner, top_p_inner)

            boxed = False
            if gen.hit_limit:
                print("HIT LIMIT")
                # In the middle of text or code, the last number is almost certainly not the answer
                result_output = -1
                # Not necessarily bad, maybe can salvage answer from code_output
            elif not bad:
                result_output, boxed = process_text_output(gen.new_output)
                if not boxed:
                    # Trying again
                    print("FORCING BOXED")
                    gen.append_prompt(r" The answer is \boxed")
                    gen.generate(0.2, top_p)
                    result_output, boxed = process_text_output(gen.decoded_output)
            
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
                    
        except subprocess.TimeoutExpired:   #DUMMY FIXME!!!!!
            pass

        #except IndexError as e: #Exception as e:
        #    print("predict() EXCEPTION")
        #    print(e)
        #    result_output = -1

        if not bad and result_output > -1:
            outputs.append((result_output, score, result_info))
            answer_scores[result_output] += score

        if len(outputs) > 0:
            answers = [(score,ans) for (ans,score) in answer_scores.items()]
            answers.sort(reverse = True)
            print("SCORES,ANSWERS:", answers)
            best_score, best = answers[0]
            #if best_score >= 3 and best_score >= 1 + (jj+1)/2:
            if best_score > 4:
                print("ANSWER FOUND!")
                break

    print("\nAll outputs:", outputs)
    return best


# %% jupyter={"outputs_hidden": false}
env = make_env()
iter_test = env.iter_test()

NOTEBOOK_START_TIME = time.time()
for probi, (test, sample_submission) in enumerate(iter_test):
    sample_submission['answer'] = predict(probi, test['problem'].values[0])
    #print(f"Making prediction for ""{test[:100]}"": {sample_submission}")
    env.predict(sample_submission)

# %% jupyter={"outputs_hidden": false}
if not PRIVATE:
    print(env.df)
    score = (env.df.ground_truth == env.df.answer).sum()
    print(f'{score} matches in {len(env.df)} examples')

# %% jupyter={"outputs_hidden": false}