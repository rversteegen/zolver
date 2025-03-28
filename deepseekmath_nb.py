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
#os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"   # Makes things very slow, doesn't help that much

import time
NOTEBOOK_START_TIME = time.time()


if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    PRIVATE = True
else:
    PRIVATE = False

SLOW = True # = PRIVATE
LOGGING = not PRIVATE
VALIDATE, DSET_TAG = "AMC_12_valid", "AMC12V"
#VALIDATE, DSET_TAG = "AIME_test24", "AIME24"
MAX_DIFFICULTY = 5

VLLM = False

if PRIVATE:
    SLOW = True
    VALIDATE = False
OLDCODE = False  # Original code runner
TB = True # Show traceback, OLDCODE=False only
ASK_CONFIDENCE = False  # "Is it proven?"
SINGLE_GPU = False   # Just 1xT4
QUANT = False
if QUANT:  #Fits on one device
    SINGLE_GPU = True


USE_PAST_KEY = True
SEED = 324
#https://www.kaggle.com/code/voxelate/deepseek-math-7b-rl
MODEL_PATH = "/kaggle/input/deepseek-math-7b-rl/deepseek-math"
MISTRAL = False
LLEMMA = False   # LLEMMA tokenizer
RELOAD_MODEL = False   # For interactive run-all
DEVICE = 'cuda' #if torch.cuda.is_available() else 'cpu'
N_REPETITIONS = 3 if VALIDATE else (25 if SLOW else 1)   # 6
MAX_SINGLE_GEN_TOKENS = 1400
MAX_GEN_TOKENS = 1600 if SLOW else 500
#MAX_TOKENS = 1500 if (P100 and USE_PAST_KEY) else 2048
MAX_TOKENS = 2048

RELOAD_MODEL2 = True
MODEL_PATH2 = '/kaggle/input/download-embeddedllm-mistral-7b-merge-14-v0-4/EmbeddedLLM-Mistral-7B-Merge-14-v0.4/'
#MODEL_PATH2 = '/kaggle/input/aimo-24-model-meta-math-metamath-mistral-7b'
QUANT2 = False

FIRSTPROB = 0  # ignored for PRIVATE

if PRIVATE:
    NPROBS = 50
    TIME_LIMIT = 32000
elif VALIDATE:
    NPROBS = 10 #100
    TIME_LIMIT = 3000
else:
    NPROBS = 1  #10
    TIME_LIMIT = 450


# %% [markdown]
# # Install and import libraries, aimo/dummy module

# %%
%%time
!pip uninstall -y fastai -qq  # Conflicts with torch>=2.3.0
!pip uninstall  -y torch -qq
!pip install  -U --no-index --find-links=/kaggle/input/pytorch-and-libs -U torch -qq #torch 2.3.1

if VLLM:
    %env MAX_JOBS = 4
    #!pip install -U /kaggle/input/bitsandbytes-0-42-0-py3-none-any-whl/bitsandbytes-0.42.0-py3-none-any.whl
    !pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    !pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl

# %%
import torch
P100 = (torch.cuda.device_count() == 1)
if P100:
    SINGLE_GPU = True

LOG_TAG = "ζv1-MMerge14+DSM"
if FIRSTPROB:
    LOG_TAG += f"{FIRSTPROB}-{FIRSTPROB + NPROBS}"
else:
    LOG_TAG += f"{NPROBS}"
if MAX_DIFFICULTY < 9:
    LOG_TAG += f"(d{MAX_DIFFICULTY})"
LOG_TAG += f"of{DSET_TAG}_Q{QUANT if QUANT else 'off'}_{MAX_GEN_TOKENS}-{MAX_SINGLE_GEN_TOKENS}tok"
if P100:
    LOG_TAG += "_P100"
elif SINGLE_GPU:
    LOG_TAG += "_1xT4"
else:
    LOG_TAG += "_2xT4"
LOG_TAG += f"{'' if USE_PAST_KEY else '_noKV'}{'' if TB else '_noTB'}{'_didprove' if ASK_CONFIDENCE else ''}"
LOG_NAME = time.strftime("%Y%m%d-%H%M-") + LOG_TAG + ".csv"
   

print(f"P100={P100}, DEVICE={DEVICE}, QUANT={QUANT}, SLOW={SLOW}")
print(LOG_NAME)

# %% _kg_hide-input=true
%%time
if (QUANT2 or VLLM or QUANT) and not 'installed_libs' in globals():
    # Need more recent accelerate
    # https://www.kaggle.com/datasets/anrenk/accelerate-0-29-3/versions/1
    # https://www.kaggle.com/datasets/anrenk/bitsandbytes-0-43-1/versions/1
    #!pip install -U /kaggle/input/accelerate-0-29-3/accelerate-0.29.3-py3-none-any.whl -qq
    !pip install -U /kaggle/input/pytorch-2-3-0-and-libs/accelerate-0.29.3-py3-none-any.whl -qq
    #!pip install -U /kaggle/input/bitsandbytes-0-43-1/bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl -qq
    !pip install -U /kaggle/input/pytorch-2-3-0-and-libs/bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl -qq
    installed_libs = True
    
!pip install -U /kaggle/input/z3-python-wheel/z3_solver-4.12.5.0-py2.py3-none-manylinux2014_x86_64.whl -qq

# %%
import sys
import os
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import re
import subprocess
import math
import random
from collections import defaultdict
from collections import Counter
import torch
import transformers
import accelerate
from torch import multiprocessing
if VLLM:
    from vllm import LLM, SamplingParams

sys.path.append('/kaggle/input/zolver/ζolve')
import llm_prompting


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
                    if self.df['difficulty'][self.counter] > MAX_DIFFICULTY:
                        self.counter += 1
                        continue
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
    if "During handling of the above exception" in output:
        # Only show last exception, which is the user-interpretable error message sympy gives
        output = output.split("During handling")[-1]
    
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
        text = match.group()
        if "real" not in text and "imaginary" not in text and "complex" not in text:
            #print("...adding real=True!")
            return text[:-1] + ', real=True)'
        else:
            return text[:-1] + ', finite=True)'
    code = "import math, fractions, sympy\nfrom sympy import *\n" + code
    code = re.sub(r"symbols\([^)]+\)", repl, code)
    # Add a try...except block
    #code = code.replace('\n', '\n    ')
    #code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)

    with open('input.py', 'w') as fout:
        fout.write(code)
    #batcmd = 'timeout 7 ' + sys.executable + ' code.py'
    
    code_status = False
    try:
        startt = time.time()
        process = subprocess.run("timeout 14 " + sys.executable + ' input.py', shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        shell_output = (process.stdout + process.stderr).decode('utf8')
        #shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        print(f"<<<<<###<Result ({time.time() - startt :.1f}s):\n" + shell_output + ">###>>>>>")

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

def shorten_overlong_lines(code_output):
    lines = []
    for line in code_output.split("\n"):
        if len(line) > 140:
            try:
                start = line.index(',', 50)
                end = line.rindex(',', 0, -50)
                if start < end:
                    line = line[: start+1] + " ... " + line[end:]
            except ValueError:
                pass
        lines.append(line)
    return '\n'.join(lines)


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
            result_output = re.findall(r' answer is:?\s*\$*(\d+)', result)  # the answer / the final answer
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
# # Util

# %%
def show_gpu_mem():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"CUDA dev {i}: free={free/2**30 :.2f} used={alloc/2**30 :.2f} + reserved={(reserved - alloc)/2**30:.2f} / {total/2**30:.1f} GB")
        
def gpu_mem():
    if PRIVATE:
        return ""
    ret = ""
    for i in range(torch.cuda.device_count()):
        reserved = torch.cuda.memory_reserved(i)
        use = torch.cuda.memory_usage(i)
        ret += f"(cuda{i}: {reserved/2**30 :.2f}GB,{use}%) "
    for i in range(torch.cuda.device_count()):
        temp = torch.cuda.temperature(i)
        use = torch.cuda.memory_usage(i)
        MHz = torch.cuda.clock_rate(i)
        ret += f"(cuda{i}: {temp}°C,{use}%,{MHz}MHz) "
    return ret

def cpu_time() -> str:
    if PRIVATE:
        return ""
    threadt = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
    processt = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
    return f"{threadt:.1f}/{processt:.1f}s CPU"

def show_model_mem(model):
    class MemUse: params, bufs = 0, 0
    counts = defaultdict(MemUse)
    for buf in model.parameters():
        #print("size", buf.numel() * buf.itemsize)
        counts[buf.device].params += buf.numel() * buf.itemsize
    print("buffers:")
    for buf in model.buffers():
        counts[buf.device].bufs += buf.numel() * buf.itemsize
    print("Model parameters+buffers:")
    for dev in counts:
        print(f"  {dev}: {counts[dev].params / 2**30 :.2f} + {counts[dev].bufs / 2**30 :.2f} GB")
    mem = model.get_memory_footprint()
    print("  Total:", "%.2f GB" % (mem / 2**30))
    

def nvidia_pstate() -> str:
    "Get string telling performance state and any active throttling or other 'Clocks Event Reasons'"
    if PRIVATE:
        return ""
    out = subprocess.check_output("nvidia-smi -q -d PERFORMANCE", shell = True).decode()

    info = ""
    gpublocks = out.split("\nGPU ")[1:]
    for i, block in enumerate(gpublocks):
        info += f"(GPU{i}:"
        for line in block.split("\n"):
            if "Performance State" in line:
                # This is always before the Active/Not Active/N/A lines
                info += 'Pstate=' + line.split(" ")[-1] + " ["
            elif ": Active" in line:
                label = line.split(":")[0].strip()
                info += f"[{label}]"
        info += "]) "
    return info

def param_size(param):
    "Size of the parameters of a Pytorch Module (e.g. a Transformer layer) in bytes"
    return param.numel() * param.itemsize

def module_size(mod):
    "Size of the parameters of a Pytorch Module (e.g. a Transformer layer) in bytes"
    return sum(param.numel() * param.itemsize for param in mod.parameters())

def move_model_params_to(model, device = 'cpu', MB = {'cuda:0':1500, 'cuda:1':1500}):
    """Move some of the modules of a Pytorch model from the devices listed in the keys of `MB` to `device`.
    Moves parameter tensors until at least the number of megabytes of memory given in `MB` have been freed.
    If called repeatedly will keep moving more.
    The model will be unusable until it's reloaded with move_model_params_back()."""
    moved = defaultdict(int)
    # Iterate .modules() rather than .parameters() because the Tensor.to() method doesn't act in-place
    #for name, mod in model.named_modules():
    for mod in model.modules():
        # This is a depth-first search, so only move leaf Modules to prevent double-counting.
        # Alternatively could restrict to modules named "model.layers.##":
        #if re.match('^lm_head|model.embed_tokens|model.layers.\d+$', name):
        if next(mod.children(), None) is None:  # Has no children
            firstparams = next(mod.parameters(), None)
            if firstparams is None:
                continue  # No parameters, e.g. a rotary_emb Module
            olddevice = str(firstparams.device)  # There isn't a mod.device attribute
            if moved.get(olddevice, 0) >= MB.get(olddevice, 0) * 2**20:
                continue
            #print(f"Moving {name} from {olddevice}")
            mod._former_device = olddevice  # Save for move_model_params_back()
            mod.to(device)
            moved[olddevice] += module_size(mod)
    print("Moved", {dev: int(amnt/2**20) for (dev,amnt) in moved.items()}, "MB to", device)

def move_model_params_back(model):
    "Undoes move_model_params_to()."
    moved = defaultdict(int)
    for name, mod in list(model.named_modules()):
        if hasattr(mod, '_former_device'):
            olddevice = str(next(mod.parameters()).device)
            #print(f"Moving {name} from {olddevice} to {mod._former_device}")
            moved[mod._former_device] += module_size(mod)
            mod.to(mod._former_device)
            del mod._former_device
    print("Moved", {dev: int(amnt/2**20) for (dev,amnt) in moved.items()}, "MB back")


# %% [markdown]
# # Load model

# %%
multiprocessing.set_start_method("spawn")

# %%
%%time
transformers.set_seed(SEED)


def load_model(MODEL_PATH, QUANT, device_map = None):
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


    if VLLM:
        model = LLM(model=MODEL_PATH, 
              tokenizer=MODEL_PATH,
              dtype="half",
              tensor_parallel_size=2,
              max_model_len=4096,
              enforce_eager=True,
              gpu_memory_utilization=0.95,
              trust_remote_code=True,
              #swap_size=7,
              seed=SEED)

        return model
    else:

        config = transformers.AutoConfig.from_pretrained(MODEL_PATH, use_cache=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

        config.gradient_checkpointing = True # Enable gradient checkpointing for memory optimization
        config.pad_token_id = tokenizer.pad_token_id

        if device_map is None:
            LAYERS_GPU0 = 32 if P100 else 15
            device_map = [('model.embed_tokens', 0)] + [(f'model.layers.{i}', 0 if i < LAYERS_GPU0 else 1) for i in range(0, 31 + 1)] + [
                             ('model.norm', 1),
                             ('lm_head', 1)]
            device_map = {ii:jj for (ii,jj) in device_map}

            if SINGLE_GPU:
                #model_kwargs['device_map'] = "auto"
                device_map = "cuda:0" #"sequential"
                #model_kwargs['device_map'] = device_map
           
        model_kwargs['device_map'] = device_map


        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            config=config,
            **model_kwargs
        )

        # Disable memory-efficient sparse tensors for CUDA operations
        #torch.backends.cuda.enable_mem_efficient_sdp(False) ######### SDP

        print()
        print("Model", MODEL_PATH)
        print("dtype", model.dtype, model.hf_device_map)

        show_model_mem(model)
        return model, tokenizer

if not 'model' in globals() or RELOAD_MODEL:
    model = None
    gc.collect()
    torch.cuda.empty_cache()
if model is None:
    model, tokenizer = load_model(MODEL_PATH, QUANT)


if not 'model2' in globals() or RELOAD_MODEL2:
    model2 = None
    gc.collect()
    torch.cuda.empty_cache()
if model2 is None:
    # FOR MISTRAL

    device_map = [('model.embed_tokens', 0)] + [(f'model.layers.{i}', 0 if i < 16 else 1) for i in range(0, 32)] + [
                     ('model.norm', 1),
                     ('lm_head', 1)]
    device_map = {ii:jj for (ii,jj) in device_map}

    model2, tokenizer2 = load_model(MODEL_PATH2, QUANT2, device_map) # if QUANT2 else "auto")
        
show_gpu_mem()

# %%
#dir(torch.backends.cuda)
torch.backends.cuda.enable_mem_efficient_sdp(True)
print(torch.backends.cuda.flash_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
print(torch.backends.cuda.mem_efficient_sdp_enabled())
#print(torch.backends.opt_einsum.is_available())
#print(torch.backends.opt_einsum.enabled)


# TEST SDPA

query = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.bfloat16, device="cuda")

res = torch.nn.functional.scaled_dot_product_attention(query,key,value)


# %%
def safe_tok_line(line, tokenizer):
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
    def __init__(self, model, tokenizer, stops, allows = [], need_safetok = LLEMMA or MISTRAL):
        super().__init__()
        
        self.allows = [tokenizer(word, return_tensors='pt', add_special_tokens=False)['input_ids'][0].to(model.device) for word in allows]
        self.stop_words = []
        self.stops = []
        for stop_word in stops:
            if need_safetok:
                toks = safe_tok_line(stop_word, tokenizer)
            else:
                toks = tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            if stop_word.endswith("DROP"):
                toks = toks[:-1]
                stop_word = stop_word.replace("DROP", "")
            self.stops.append(toks.to(model.device))
            self.stop_words.append(stop_word)
        self.ignore_count = 0

    def __call__(self, input_ids: torch.LongTensor, _scores):
        for i, stop in enumerate(self.stops):
            suffix = input_ids[0][-len(stop):]
            if torch.all(torch.eq(stop, suffix)):
                #print("found", repr(self.stop_words[i]), self.ignore_count)
                for j, allow in enumerate(self.allows):
                    suffix = input_ids[0][-len(allow):]
                    if torch.all(torch.eq(allow, suffix)):
                        print("  ignoring allow word!")
                        break
                else:
                    if self.ignore_count <= 0:
                        return True
                    print("  ignoring stop!")
                    self.ignore_count -= 1
        return False
    
class NGramStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, ngram_size, spacing = 4):
        super().__init__()
        self.ngram_size = ngram_size
        self.spacing = spacing
        self.reset()
        
    def reset(self):
        self.ngrams = set()
        self.inside_code = False
        self.signalled = False
        self.startlen = 0
        
    def __call__(self, input_ids: torch.LongTensor, _scores):
        assert _scores is None
        if self.inside_code:
            return False
        if len(input_ids[0]) <self.ngram_size: #  + self.startlen + 
            # Too close to the start of this text block, contains part of previous code/output
            return False
        suffix = tuple(input_ids[0][-self.ngram_size::self.spacing].tolist())
        #if "```" in suffix:
        if suffix in self.ngrams:
            self.signalled = True
            return True
        self.ngrams.add(suffix)
        return False
    

BEFORE_DOCSTRING = "():\n   "  # 3 spaces! A 4th space is part of the next token!

# ```->[10897], ````->[4885, 4885], `````->[4885, 10897], ``````->[4885, 4885, 4885]
stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , ")\n\n```" , "```\n", "````"] #, BEFORE_DOCSTRING]
main_stoplist = StoppingCriteriaSub(model, tokenizer, stop_words)
error_stoplist = StoppingCriteriaSub(model, tokenizer, [" error", " mistake"], [" trial and error"])
ngram_stopper = NGramStoppingCriteria(130)

def deepseek_stoppers():
    stopping_criteria = transformers.StoppingCriteriaList([ngram_stopper, main_stoplist])
    error_stopping_criteria = transformers.StoppingCriteriaList([ngram_stopper, main_stoplist, error_stoplist])
    
    return stopping_criteria, error_stopping_criteria

def other_stoppers():
    stop_words2 = ["```", "```\n", "````"]
    stoplister2 = StoppingCriteriaSub(model2, tokenizer2, stop_words2, need_safetok =True)
    stopping_criteria2 = transformers.StoppingCriteriaList([stoplister2])
    return stopping_criteria2, stopping_criteria2

#torch.cuda.empty_cache()
#gc.collect()

class GenStreamWatcher(transformers.generation.streamers.BaseStreamer):
    def __init__(self):
        self.start_time = None
        self.tokens = []
        
    def put(self, token_ids):
        # First the prompt is fed in, before KVs are built for it.
        if list(token_ids.shape) == [1]:
            self.tokens.append(token_ids.tolist()[0])
            #self.tokens.append(token_ids[0])
            if self.start_time is None:
                #print("stream start")
                self.start_time = time.time()
            #print(tokenizer.decode(token_ids), end='')
        
    def end(self):
        pass


# %% [markdown]
# # Prompts

# %% _kg_hide-input=false
#NOT 3-TUPLE
code = ("""Dear professor, consider this math problem:
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions. Your final answer should be non-negative integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.""",
"Approach:")
#NOT 3-TUPLE
code2 = ("""Consider this math problem:
\"{}\"
First, logically analyze the implications of the problem statement. Second, list the general steps of a Sympy-based approach to calculate the answer. Third, write out commented Sympy code to compute the numerical answer and print the result.
You can run and receive results of multiple code blocks to reach the answer in stages. 
Note that intermediate calculations may be real numbers.
Finally, output the final integer answer (not an algebraic expression) within \\boxed{{}}.""",)

#NOT 3-TUPLE
cot = ("""Below is a math problem you are to solve:
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final integer answer within \\boxed{{}}.""",)

prompt_options = [code2, code, cot]

# Original prompts

# You can run multiple code blocks to reach the answer in steps.

#similar to 'code' in V9 soln, but several changes inc "run multiple code blocks" (changes were in V11?)
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

# cot0 == cot in V9 and V11
# cot1 replaces 'numerical' with 'integer'
cot1 = ("cot1", """Below is a math problem you are to solve (positive integer answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. """
"""After solving the problem, output the final integer answer within \\boxed{{}}.""",
   ""    )

cot2 = ("cot2", """Below is a math problem you are to solve (non-negative integer answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. """
"""Output the final answer within \\boxed{{}}.""",
 "")

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

compute0 = ("compute0", """{}
The answer is a non-negative integer.
\nFind the solution by computing every step along the way with python/sympy/numpy/scipy/etc. Don't state solutions without either proof or code but don't repeat or restate code outputs.
Put your final answer within \\boxed{{}}.""",
"")

# Not as good
compute1 = ("compute1", """{}
The answer is a non-negative integer.

Show the solution by computing every step along the way and every algebraic simplification with python/sympy and printing results. Don't jump to conclusions without either proof or code.
Put your final answer within \\boxed{{}}.""",
"")


cot2 = ("cot2",  """Below is a math problem you are to solve (positive integer answer!):
\"{}\"

Write an efficient python program to solve it. Write out the whole program and print the result so it will run. If it doesn't work, don't try the same thing repeatedly. Be concise. Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "Our approach")


cot3 = ("shortpkg0", """Here's a problem, with a positive integer answer!
\"{}\"
Analyze step by step and use python/sympy/numpy/scipy/etc to do any calculations or find solutions. After solving the problem, output the final integer answer within \\boxed{{}}.""",
       "")

# analy0 == code2 in V9 and V11
# analy1 adds "No docstrings."
# analy2 removes No docstrings and adds non-negative integer, rewords
analy1 = ("analy1",
    """Consider this math problem:
\"{}\"
First, analyze the implications of the problem statement and restate it more mathematically. Write code to check assumptions, to simplify the problem.
Write out commented Sympy code to compute the numerical answer and print the result. But no docstrings.
Think step by step to come to a solution with programs. After solving the problem, output the final integer answer within \\boxed{{}}.""",
        "")
analy2 = ("analy2",
    """Consider this math problem:
\"{}\"
The answer is a non-negative integer. First, analyze the implications of the problem statement and restate it more mathematically. Write code to check assumptions, to simplify the problem.
Write out commented Sympy code to compute the numerical answer and print the result.
Think step by step to come to a solution with programs, and put your final integer answer within \\boxed{{}}.""",
        "")



# Badly behaved, just produces code without thought 
# steps 0 had "positive integer", fixed in steps0_
steps0 = ("concise-steps0_",  """\"{}\"

Think step by step writing python code to solve this problem. Get to the point. Maths only, no chatting with me. Write out the whole program and print the result.
If it doesn't work and you can't fix it then stop. Put your final answer within \\boxed{{}}. It must be a non-negative integer.""",
"\n"        )  # Hopefully stops it producing pythogn without ````python

steps1 = ("concise-steps1",  """\"{}\"

Think step by step writing python code to solve this problem. The answer is a non-negative integer. Get to the point. Maths only, no chatting with me. Write out the whole program and print the result.
Don't repeat yourself: if you keep making the same mistake then stop. Put your final answer within \\boxed{{}}.""",
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
#prompt_options = [steps0, tool0, tool1]  # 22
prompt_options = [steps1, tool0, tool1]  # 17
prompt_options = [elab0tool, tool0, tool1]  # 18
prompt_options = [elab0tool, tool0, steps1]  # 15, 17, ?19

prompt_options = [compute0, elab0tool] 
prompt_options = [compute0, tool0, analy2] # 16
prompt_options = [compute0, tool0, tool1, steps1, analy2]  #, elab0tool, ]
prompt_options = [tool1, analy2] # 21, 17, X16
prompt_options = [elab0tool, tool1]
#prompt_options = [tool1]
#prompt_options = [tool1, elab0tool, compute0]   # for validation

# %% [markdown]
# # Generation and main loop

# %%
# It grew out of control...

class LLMGenerator:
    def __init__(self, model, tokenizer, first_model = True):
        self.model = model
        self.tokenizer = tokenizer
        self.tokens = self.tokenizer("")['input_ids']
        self.prompt = self.tokenizer.decode(self.tokens)  # Will be equal to tokenizer.bos_token
        self.num_gen_tokens = 0  # Total, not including prompt and outputs
        self.past_key_values = None
        self.hit_limit = False
        self.need_update = True
        self.stop_on_error = True
        self.bad_words = None
        self.first_model = first_model
        deepseek = first_model
        self.deepseek = deepseek
        if deepseek:
            ngram_stopper.reset()
            self.stoppers = deepseek_stoppers()
        else:
            self.stoppers = other_stoppers()

    def update_inputs(self):
        if True:
            # split_special_tokens = False is the default, so <｜end▁of▁sentence｜> will convert
            self.model_inputs = self.tokenizer(self.prompt, add_special_tokens = False, return_tensors='pt').to(self.model.device)
        else:
            self.model_inputs = self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        self.tokens = self.model_inputs['input_ids'][0]
        self.decode_tokens()   # Fix off by one bug in .new_output
        
    def decode_tokens(self):
        self.prompt = self.tokenizer.decode(self.tokens, skip_special_tokens = False)
        self.need_update = False
        
    def user_prompt(self, text, assist_prefix = "", show = True):
        """Add `text` to the prompt as a User message, followed by Assistant:, and optionally start of assistant response.
        Call this before the first .generate() or at any time to add a User direction."""
        old_input = len(self.tokens)
        ol = len(self.prompt)
        if self.deepseek:
            # DeepSeek specific template
            if old_input > 1 and not self.prompt.endswith(self.tokenizer.eos_token):
                self.prompt += self.tokenizer.eos_token
            if True:
                self.prompt += ("User: " + text.strip() + "\n\nAssistant: " + assist_prefix.strip(' ')).strip(' ')
            else:
                self.prompt += ("User: " + text.strip() + "\n\n" + assist_prefix.strip(' ')).strip(' ')
        elif True:
            # Not implemented
            self.prompt += ("### Instruction:\n" + text.strip() + "\n\n### Response:\n" + assist_prefix.strip(' ')).strip(' ')
        else:
            self.messages = [{"role": "user", "content": text}]
            if assist_prefix:
                self.messages.append({"role": "assistant", "content": assist_prefix})
            self.model_inputs = self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        self.update_inputs()
        if show:
            print(f"<<<<<PROMPT {len(self.tokens) - old_input} tokens ({len(self.tokens)} total)\n" + self.prompt[ol:] + ">>>>>")

    def append_prompt(self, text, show = True):
        "Append to prompt, without changing the role."
        text = text.rstrip(' ')
        self.prompt += text
        old_input = len(self.tokens)
        self.update_inputs()
        if show:
            print(f"<<<<<APPEND {len(self.tokens) - old_input} tokens ({len(self.tokens)} total)\n" + text + ">>>>>")
        
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

    def generate(self, temperature = 0.9, top_p = 1.0, limit = 9999, show = True, skip_check = False):
        startt = time.time()
        if skip_check:
            max_toks = limit
        else:
            max_toks = self.check_limit()
            if max_toks < 3:
                print("SKIP GENERATE DUE TO LIMIT")
                self.new_output = ""
                return

        #if self.need_update:
        #    self.update_inputs()  # Regenerate model_input
        if self.stop_on_error:
            stopper = self.stoppers[1] #error_stopping_criteria
        else:
            stopper = self.stoppers[0] #stopping_criteria

        streamer = None  #GenStreamWatcher()
        
            
        input_len = len(self.tokens)
        #print("ngram stop inactive=", ngram_stopper.inside_code)
        try:
            generation_output = self.model.generate(**self.model_inputs, 
                                               max_new_tokens = min(limit, max_toks),
                                               return_dict_in_generate = USE_PAST_KEY,
                                               past_key_values = self.past_key_values,
                                               do_sample = True,
                                               temperature = temperature,
                                               top_p = top_p,
                                               bad_words_ids = self.bad_words,
                                               #no_repeat_ngram_size = 50,
                                               pad_token_id = self.tokenizer.eos_token_id,
                                               num_return_sequences = 1,
                                               streamer = streamer,
                                               stopping_criteria = stopper)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e: # RuntimeError if pytorch caching allocator disabled
            print(e)
            if streamer is None:
                raise
            if len(streamer.tokens) == 0:
                print("OOM: tried to recover but no tokens generated")
            self.hit_limit = True
            self.past_key_values = None
            self.tokens = torch.tensor(self.tokens.tolist() + streamer.tokens)
            print("OOM recovered")
            self.model_inputs = {'input_ids': self.tokens}  # torch.tensor([self.tokens]).to(self.model.device)
            gentime = time.time() - startt
        else:
            if USE_PAST_KEY:
                sequences = generation_output.sequences
                self.past_key_values = generation_output.past_key_values
            else:
                sequences = generation_output


            self.tokens = sequences[0]
            self.model_inputs = {'input_ids': sequences}
            if streamer:
                assert sequences[0][input_len:].tolist() == streamer.tokens  # Checked it passes, so now leave commented
                gentime = time.time() - streamer.start_time
            else:
                gentime = time.time() - startt

        decoded_output = self.tokenizer.decode(self.tokens, skip_special_tokens = False)
        self.new_output = decoded_output[len(self.prompt):]
        self.prompt = decoded_output
        self.need_update = True

        #self.new_output = self.tokenizer.decode(gen.tokens[input_len:], skip_special_tokens=True)

        runt = time.time() - startt

        new_toks = len(self.tokens) - input_len
        self.num_gen_tokens += new_toks
        if not skip_check:
            if new_toks >= MAX_SINGLE_GEN_TOKENS:
                print("HIT MAX_SINGLE_GEN_TOKENS")
                self.hit_limit = True
            if self.num_gen_tokens >= MAX_GEN_TOKENS:
                print("HIT MAX_GEN_TOKENS")
                self.hit_limit = True   # TODO: instead append  "...\nPutting that into code:\n```python"
            if len(self.tokens) >= MAX_TOKENS:
                print("HIT MAX_TOKENS")
                self.hit_limit = True

        if show:
            print(f"<<<<<GEN {new_toks} tokens ({len(self.tokens)} total) in {gentime :.2f}+{runt - gentime :.2f}s ({(new_toks-1)/gentime :.2f} tok/s) ({cpu_time()}) {gpu_mem()} {nvidia_pstate()}\n"
                  + self.new_output
                  + ">>>>>")

    def endswith(self, text):
        return self.prompt.endswith(text)
    
    def set_bad_words(self, words: list):
        self.bad_words = self.tokenizer(words, add_special_tokens = False)['input_ids']


# %%
import ζ.util
from llm_prompting import AnswerLog

if LOGGING:
    AnswerLog.log_path = LOG_NAME

def predict(probi, problem):

    temperature = 0.75
    top_p = 0.95
    temperature_coding = 0.75
    top_p_coding = 0.95

    firstprompt = random.randint(0, 9999)
    score = 0
    best = 0

    time_left = TIME_LIMIT - (time.time() - NOTEBOOK_START_TIME)
    time_for_item = time_left / max(1, NPROBS - probi)
    item_time_start = time.time()

    answerlog = AnswerLog(env.prob_id)
    
    # ζolve
    if True:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            # Remove part of first model (DeepSeek-Math-7b-rl) from GPU, only takes few seconds
            with ζ.util.Timer() as timing:
                move_model_layers_to(model, 'cpu', {'cuda:0':3900, 'cuda:1':3900})
            with ζ.util.Timer() as timing:
                move_model_layers_back(model2)
            print("Moved in", timing)
            torch.cuda.empty_cache()
            gc.collect()

            def model2makegen():
                return LLMGenerator(model2, tokenizer2, first_model = False)

            ζol = llm_prompting.ζolver(problem, answerlog)
            #ζol = ζolver(problem)
            solved = ζol.doit(model2, tokenizer2, model2makegen, timeout = min(time_for_item, 300), hard_timelimit = NOTEBOOK_START_TIME + TIME_LIMIT)
            print(f"{int(time.time() - item_time_start)}s spent in ζolve()")
            best = ζol.best
            if solved:
                return ζol.best
            
            # Fix occasional OOM
            move_model_layers_to(model2, 'cpu', {'cuda:0':800, 'cuda:1':800})
        except RuntimeError:# as e:#Exception as e:
            print("UNCAUGHT EXCEPTION!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)

    return best

    move_model_layers_back(model)

    for jj in range(N_REPETITIONS): # tqdm(range(N_REPETITIONS)):

        #temperature = 0.9 * (0.3+0.25*jj) / (0.25*jj + 1)  # 0.27 0.40 0.48 0.54 0.59 0.62 0.65 ... 0.71 ... ?
        temperature = (0.35+0.25*jj) / (0.25*jj + 1)   # 0.35 0.48 0.57 .... 0.8 ... 0.89
        #temperature = (0.5+0.15*jj) / (0.15*jj + 1)   # 0.5 0.56 ... 0.79... 0.87
        temperature_coding = temperature
        
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

        logrow = answerlog.new_row()
        
        last_code_error = None
        last_code_output = None
        code_repeat_count = 0
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
            print(psutil.cpu_stats())
            print(psutil.cpu_times())
            print()
            
            gen = LLMGenerator(model, tokenizer, True)
            gen.set_bad_words([' """', " '''", '():\n    r'])  # disallow  r""" too

            promptname, prompt, assist = prompt_options[(firstprompt+jj) % len(prompt_options)]  #random.choice(prompt_options)
            print("prompt", promptname)
            logrow.prompt = promptname
            gen.user_prompt(prompt.format(problem), assist)
            gen.generate(temperature, top_p)
            
            currently_text = True
            temperature_inner = temperature
            top_p_inner = top_p

            while not gen.hit_limit and gen.check_limit() > 3:
                continue_block = False
                
                if re.search("Assistant: (from|import) ", gen.prompt):   #gen.new_output.startswith(" from "):
                    # ``` is missing
                    print("FIX INITIAL MISSING ```")
                    gen.replace_prompt(gen.prompt.replace("Assistant: ", "Assistant: ```python\n"))
                    currently_text = False
                
                if ngram_stopper.signalled:
                    print("NGRAM STOP")
                    bad = True
                    break
                
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
                    if False: ##gen.endswith("trial and error"):
                        print("...IGNORE ERROR")
                        error_stoplist.ignore_count += 1
                        continue_block = True
                    else:
                        print("MISTAKE")
                        bad = True
                        break
                    
                elif not any(gen.endswith(stop_word) for stop_word in main_stoplist.stop_words):
                    # No continuation, and didn't hit limit, so must have ended due to eos.
                    # This should never happen!
                    result_info = "no continue"
                    print("nocont", repr(gen.prompt[-10:]))
                    break


                if gen.endswith(BEFORE_DOCSTRING):
                    # Oh you think you're going to copy the problem into the docstring, do you? No!!!
                    gen.append_prompt(" #")
                    continue_block = True


                if continue_block:
                    pass
                # Model sometimes outputs a line of ```` often not ending in 'python'
                elif gen.endswith("```python") or (currently_text and gen.endswith("````") ):
                    # Starting code
                    print("CONSIDER IT")
                    if code_blocks == 3:
                        result_info = "toomuchcode"
                        break
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
                            if not code_status:
                                logrow.code_errors += 1

                            if code_status == True:
                                code_error_count = 0
                                cumulative_code += code_text
                                new_len = len(code_output)
                                code_output = code_output[prev_code_len:].strip()
                                prev_code_len = new_len
                                print("CODE new_len =", new_len)
                                if code_output and code_output == last_code_output:
                                    code_repeat_count += 1
                                else:
                                    code_repeat_count = 1
                                last_code_output = code_output
                                if code_repeat_count >= 3:  # Same output three times
                                    # Not bad
                                    print("REPEATED OUTPUT")
                                    result_info = "repoutput"
                                    break
                                try:
                                    last_code_result = round(float(eval(code_output.strip().split("\n")[-1])))
                                except:
                                    pass
                                code_output = shorten_overlong_lines(code_output)

                            else:
                                # the last line is the exception
                                except_line = code_output.strip().split("\n")[-1]
                                if except_line == last_code_error:
                                    code_error_count += 1
                                else:
                                    code_error_count = 1
                                last_code_error = except_line
                                if code_error_count >= 2:  # Same error twice
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
                    ngram_stopper.startlen = len(gen.tokens)

                    # if code_status:
                    #     #if gen.endswith(")\n```"):
                    #     if gen.endswith("\n```"):
                    #         gen.append_prompt('```output\n' + code_output + '\n```\n')
                    #     else:
                    #         #gen.append_prompt('\n' + code_output + '\n```\n')
                    #         gen.append_prompt('\n>>> ' + code_output + '\n\n')
                    # else:
                    #     pass #cumulative_code = ""

                if continue_block:
                    pass
                elif code_status == False:
                    error_stoplist.ignore_count += 1
                    #gen.stop_on_error = False   # Don't stop on "error", the LLM may say "to fix the error"
                else:
                    gen.stop_on_error = True
                ngram_stopper.inside_code = not currently_text
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
                    penalty = 0.05
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
                if False and gen.check_limit() > 50:
                    gen.user_prompt("Which step of the above solution are you least sure about? And how uncertain is it on a scale from 1 (safe) to 10 (unsound)?")
                    gen.generate(0.8, top_p, 400)
                elif ASK_CONFIDENCE and penalty == 0 and boxed and gen.check_limit() > 23:
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
            if 1 <= code_blocks: # <= 3:
                score += 0.2
            score -= penalty
                    
        #except torch.cuda.OutOfMemoryError as e:  # or RuntimeError if pytorch caching allocator disabled
        except Exception as e:   
            print("predict() EXCEPTION")
            print(e)
            logrow.exception(e)
            #result_output = -1
            bad = 2
        
        if LOGGING:
            logrow.result_info = result_info
            logrow.gen_tokens = gen.num_gen_tokens
            logrow.code_blocks = code_blocks
            logrow.score = score
            logrow.answer = result_output
            logrow.time = time.time() - it_start_time
            logrow.bad = bad
            answerlog.log()

        if result_output > -1:  # and not bad
            best, best_score, score_gap = answerlog.add_answer(result_output, score, result_info)

            #if best_score >= 3 and best_score >= 1 + (jj+1)/2:
            #if best_score > 4 and not VALIDATE:
            if (score_gap >= 3 or best_score >= 6) and not VALIDATE:
                print("EARLY FINISH!")
                break

    print("\nAll outputs:", answerlog.outputs)
    return best


######################################################

env = make_env()
iter_test = env.iter_test()

if not PRIVATE:
    NOTEBOOK_START_TIME = time.time()
for probi, (test, sample_submission) in enumerate(iter_test):
    transformers.set_seed(SEED)
    sample_submission['answer'] = predict(probi, test['problem'].values[0])
    #print(f"Making prediction for ""{test[:100]}"": {sample_submission}")
    #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True): #, enable_cudnn=True ):
    env.predict(sample_submission)

# %%
if not PRIVATE:
    if LOGGING:
        with open("prompts.txt", "w") as fo:
            fo.write(repr(prompt_options))
    print(env.df)
    score = (env.df.ground_truth == env.df.answer).sum()
    print(f'{score} matches in {len(env.df)} examples')

# %%
!rm -f LOGFILE.zip
!zip LOGFILE.zip "$LOG_NAME"

# %%
if False:
    show_model_mem(model)
    show_gpu_mem()
    model.cpu_offload()
    show_model_mem(model)
    show_gpu_mem()

# %%
if False:
    import importlib

    # make changes to example.py file
    import llm_util
    importlib.reload(llm_util)

