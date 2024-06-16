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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
# pip install cpu cores
%env MAX_JOBS = 4

!pip uninstall -y torch
!pip install --no-index --find-links=/kaggle/input/vllm-whl -U vllm

# %%
import pandas as pd
from tqdm import tqdm
PRIVATE = True

df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/test.csv')
df.head()

# %%
if len(df) < 5:
    df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/train.csv')
    PRIVATE = False
df.head()

# %% [markdown]
# # All in one code

# %%
import pandas as pd
from tqdm import tqdm
import re
import sys
import subprocess
import numpy as np
from collections import defaultdict, Counter
from numpy.random import choice
import gc
import time
import torch
from vllm import LLM, SamplingParams

# Load the dataset
df = pd.read_json('/kaggle/input/valdataset/val_67.json')
df.rename(columns={'index': 'id'}, inplace=True)
df.head()

# Define the number of repetitions
n_repetitions = 64

# Define the naive_parse function
def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
    out = reversed(out)
    return ''.join(out)

# Define the process_code function
def return_last_print(output, n):
    lines = output.strip().split('\n')
    if lines:
        return lines[n]
    else:
        return ""

def process_code(code, return_shell_output=False):
    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    code = re.sub(r"symbols\([^)]+\)", repl, code)
    if return_shell_output:
        code = code.replace('\n', '\n    ')
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
            if return_value == 'FAIL':
                CODE_STATUS = False
                return_value = return_last_print(shell_output, -2)
                if "not defined" in return_value:
                    return_value += '\nTry checking the formatting and imports'
            else:
                CODE_STATUS = True
            return return_value, CODE_STATUS  
        code_output = round(float(eval(return_value))) % 1000
    except Exception as e:
        print(e, 'shell_output')
        code_output = -1
    if return_shell_output:
        if code_output == -1:
            CODE_STATUS = False
        else:
            CODE_STATUS = True
        return code_output, CODE_STATUS  
    return code_output

# Define the process_text_output function
def process_text_output(output):
    result = output    
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', result)
        print('BOXED', result_output)
        if not len(result_output):
            result_output = naive_parse(result)
        else:
            result_output = result_output[-1]
        print('BOXED FINAL', result_output)
        if not len(result_output):
            result_output = -1
        else:
            result_output = round(float(eval(result_output))) % 1000
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    return result_output

# Define prompts
code = """Below is a math problem you are to solve (positive numerical answer):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.
•
Approach:"""

cot = """Below is a math problem you are to solve (positive numerical answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{}.\n\n"""

promplt_options = [code, cot]

# Define necessary variables for model and sampling
model_dir = '/kaggle/input/deepseek-math/'
llm = LLM(model=model_dir, 
          tokenizer=model_dir,
          dtype="half",
          tensor_parallel_size=2,
          max_model_len=4096,
          enforce_eager=True,
          gpu_memory_utilization=0.95,
          trust_remote_code=True,
          swap_size=7,
          seed=42)

tool_instruction = '\n\nPlease integrate natural language reasoning with programs to solve the above problem, and put your final numerical answer within \\boxed{}.\nNote that the intermediary calculations may be real numbers, but the final numerical answer would always be an integer.'

temperature = 0.9
top_p = 1.0

temperature_coding = 0.9
top_p_coding = 1.0

total_results = {}
total_answers = {}
best_stats = {}
total_outputs = {}
question_type_counts = {}
starting_counts = (2, 3)

# Define the main loop
NOTEBOOK_START_TIME = time.time()
TIME_LIMIT = 32000 # Define your time limit in seconds

for jj in tqdm(range(n_repetitions)):
    for I in tqdm(range(len(df))):
        TIME_SPENT = time.time() - NOTEBOOK_START_TIME
        if TIME_SPENT > TIME_LIMIT:
            break
        id_ = df['id'].loc[I]
        problem = df['question'].loc[I]
        print(f"\n\n\nQUESTION {I} - {jj} - TIME_SPENT : {TIME_SPENT:.0f} secs")
        best, best_count = best_stats.get(I, (-1, -1))
        if best_count > np.sqrt(jj):
            print("SKIPPING CAUSE ALREADY FOUND BEST")
            continue
        outputs = total_outputs.get(I, [])
        text_answers, code_answers = question_type_counts.get(I, starting_counts)
        results = total_results.get(I, [])
        answers = total_answers.get(I, [])
        for _ in range(5):
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)
        try:
            ALREADY_GEN = 0
            code_error = None
            code_error_count = 0
            code_output = -1
            counts = np.array([text_answers, code_answers])
            draw = choice(promplt_options, 1, p=counts/counts.sum())
            initail_message = draw[0].format(problem, "{}")
            prompt = f"User: {initail_message}"
            current_printed = len(prompt)
            print(f"{jj}_{prompt}\n")
            sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
            generation_output = llm.generate(prompt, sampling_params=sampling_params)
            
            # Add debugging statement
            print("Generation output type:", type(generation_output))
            print("Generation output:", generation_output)
            
            # Adjust this line based on the actual return type
            if isinstance(generation_output, list) and len(generation_output) > 0:
                if hasattr(generation_output[0], 'outputs') and len(generation_output[0].outputs) > 0:
                    decoded_output = generation_output[0].outputs[0].text
                else:
                    raise ValueError("Expected 'outputs' attribute with a nested 'text' attribute in the first element of the list")
            elif isinstance(generation_output, dict) and 'text' in generation_output:
                decoded_output = generation_output['text']
            else:
                raise ValueError("Unexpected generation output format")
            
            print(f"{decoded_output[current_printed:]}\n")
            current_printed += len(decoded_output[current_printed:])
            cummulative_code = ""
            stop_word_cond = False
            stop_words = ["```\n"]
            for stop_word in stop_words:
                stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):] == stop_word)
            while stop_word_cond and (ALREADY_GEN < TOTAL_TOKENS):
                if decoded_output[-len("```python"):] == "```python":
                    temperature_inner = temperature_coding
                    top_p_inner = top_p_coding
                    prompt = decoded_output
                else:
                    temperature_inner = temperature
                    top_p_inner = top_p
                    try:
                        if decoded_output[-len("``````output"):] == "``````output":
                            code_text = decoded_output.split('```python')[-1].split("``````")[0]
                        else:
                            code_text = decoded_output.split('```python')[-1].split("```")[0]
                        cummulative_code += code_text
                        code_output, CODE_STATUS = process_code(cummulative_code, return_shell_output=True)
                        print('CODE RESULTS', code_output)
                        if code_error == code_output:
                            code_error_count += 1
                        else:
                            code_error = code_output
                            code_error_count = 0
                        if not CODE_STATUS:
                            cummulative_code = cummulative_code[:-len(code_text)]
                            if code_error_count >= 1:
                                print("REPEATED ERRORS")
                                break
                    except Exception as e:
                        print(e)
                        print('ERROR PARSING CODE')
                        code_output = -1
                    if code_output != -1:
                        if decoded_output[-len(")\n```"):] == ")\n```":
                            prompt = decoded_output + '```output\n' + str(code_output) + '\n```\n'
                        else:
                            prompt = decoded_output + '\n' + str(code_output) + '\n```\n'
                    else:
                        prompt = decoded_output
                        cummulative_code = ""
                sampling_params = SamplingParams(temperature=temperature_inner, top_p=top_p_inner)
                generation_output = llm.generate(prompt, sampling_params=sampling_params)
                
                # Add debugging statement
                print("Generation output type:", type(generation_output))
                print("Generation output:", generation_output)
                
                # Adjust this line based on the actual return type
                if isinstance(generation_output, list) and len(generation_output) > 0:
                    if hasattr(generation_output[0], 'outputs') and len(generation_output[0].outputs) > 0:
                        decoded_output = generation_output[0].outputs[0].text
                    else:
                        raise ValueError("Expected 'outputs' attribute with a nested 'text' attribute in the first element of the list")
                elif isinstance(generation_output, dict) and 'text' in generation_output:
                    decoded_output = generation_output['text']
                else:
                    raise ValueError("Unexpected generation output format")
                
                current_printed += len(decoded_output[current_printed:])
                stop_word_cond = False
                for stop_word in stop_words:
                    stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):] == stop_word)
            raw_output = decoded_output
            result_output = process_text_output(raw_output)
            try:
                code_output = round(float(eval(str(code_output)))) % 1000  # Ensure code_output is converted to string
            except Exception as e:
                print(e, 'final_eval')
                code_output = -1
        except Exception as e:
            print(e, "5")
            result_output, code_output = -1, -1
        if code_output != -1:
            outputs.append(code_output)
            code_answers += 1
        if result_output != -1:
            outputs.append(result_output)
            text_answers += 1
        if len(outputs) > 0:
            occurances = Counter(outputs).most_common()
            print(occurances)
            if occurances[0][1] > best_count:
                print("GOOD ANSWER UPDATED!")
                best = occurances[0][0]
                best_count = occurances[0][1]
            if occurances[0][1] > 5:
                print("ANSWER FOUND!")
                break
        results.append(result_output)
        answers.append(code_output)
        best_stats[I] = (best, best_count)
        question_type_counts[I] = (text_answers, code_answers)
        total_outputs[I] = outputs
        total_results[I] = results
        total_answers[I] = answers
        print("code_answers", code_answers - starting_counts[1], "text_answers", text_answers - starting_counts[0])
        DEBUG = False
        if DEBUG:
            break

# %%
total_answers

# %%
for ii in range(len(df)):
    a = total_answers[ii]
    b = total_answers[ii]
    a = np.array(a)
    b = np.array(b)
    print(a,b)
    a[a < 0] = b[a < 0]

    pred = Counter(a.tolist()).most_common(2)
    print(pred)

# %%
import numpy as np
if PRIVATE:
    for ii in range(len(df)):
        a = total_answers[ii]
        b = total_answers[ii]
        a = np.array(a)
        b = np.array(b)
        print(a,b)
        a[a < 0] = b[a < 0]

        pred = Counter(a.tolist()).most_common(2)
        print(pred)

# %%
if PRIVATE:
    df['answer'] = [best_stats[ii][0] for ii in range(len(df))]
else:
    df['answer'] = 2

# %%
df[['id','answer']].to_csv("submission.csv", header=True, index=False)

# %%
if not PRIVATE:
    df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/train.csv')
    if PRIVATE:
        df['model_answer'] = [best_stats[ii][0] for ii in range(len(df))]
        df['match'] = df.answer == df.model_answer
        print(f'{df.match.sum()} matches in {len(df)} examples')

# %%



# %%
