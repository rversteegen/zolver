import time
import torch
import transformers

need_safetok = True


def safe_tok_line(line, tokenizer):
    """Try to make stop token sequence robust to preceding text. Works for LLemma, Mistral. InternLM2-Math seems well behaved
    Removes the leading newline."""
    #print(repr(line))
    remove_nl = False
    if not line.startswith("\n"):
        line = "\n" + line
        remove_nl = True
    nl_tok = 13  # LLEMMA and MISTRAL. 364 for InternLM2-Math
    assert tokenizer("\n", add_special_tokens = False)['input_ids'][-1] == 13
    toks = tokenizer(line, add_special_tokens = False)['input_ids']
    start = toks.index(13) + (1 if remove_nl else 0)
    #print(line, toks, "->",toks[start:])
    return torch.tensor(toks[start:])


class MultiStoppingCriterion(transformers.StoppingCriteria):
    def __init__(self, model, tokenizer, stops, need_safetok = True):
        super().__init__()

        self.stop_text = stops
        if need_safetok:
            stoplists = [safe_tok_line(stop_word, tokenizer) for stop_word in stops]
        else:
            stoplists = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'][0] for stop_word in stops]
        self.stops = [stop.to(model.device) for stop in stoplists]
        self.ignore_count = 0
        self.reset()

    def reset(self):
        self.stopped = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)
        for seqi,sequence in enumerate(input_ids):
            if seqi in self.stopped:
                is_done[seqi] = True
                continue
            for i, stop in enumerate(self.stops):
                suffix = sequence[-len(stop):]
                if torch.all(torch.eq(stop, suffix)):
                    #print("found", repr(self.stop_text[i]), self.ignore_count)
                    if self.ignore_count <= 0:
                        self.stopped[seqi] = len(sequence) - len(stop)
                        is_done[seqi] = True
                        #print("stopping", seqi)
                        break
                    print("  ignoring stop!")
                    self.ignore_count -= 1
        #return len(self.stopped) == len(input_ids)
        return is_done


def run_llm(model, tokenizer, prompt, max_tokens = 800, numseqs = 1, temp = 0.9, stopwords = ["```"], trim_stop = True, show = False):
    """Returns a list of generations.
    trim_stop: whether to include the final stopword. Either way, nothing is after it."""
    print("gen")
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    prompttoks = len(inputs['input_ids'][0])
    info = { 'input_len': prompttoks }
    startt = time.time()
    top_p = 0.96

    stopper = None
    stopperlist = None
    if stopwords:
        stopper = MultiStoppingCriterion(model, tokenizer, stopwords)
        stopperlist = transformers.StoppingCriteriaList([stopper])

    output = model.generate(**inputs,
                            max_new_tokens = max_tokens,
                            pad_token_id = tokenizer.eos_token_id,
                            do_sample = True,
                            temperature = temp,
                            top_p = top_p,
                            num_return_sequences = numseqs,
                            stopping_criteria = stopperlist)

    gentime =  time.time() - startt

    # Sequences are filled after the stopword with the padding token, which will be trimmed by skip_special_tokens
    if trim_stop and stopper:
        # Trim each sequence separately, and while at it trim the prompt tokens too
        out_text = []
        for seqi, outseq in enumerate(output):
            stop_pos = stopper.stopped.get(seqi, len(outseq))
            trimmed_seq = outseq[prompttoks : stop_pos]
            out_text.append(tokenizer.decode(trimmed_seq, skip_special_tokens=True))
    else:
        out_text = tokenizer.batch_decode(output[:, prompttoks:], skip_special_tokens=True)
        #out_text = [text[len(prompt):] for text in out_text]
    if show:
        for text in out_text:
            print("-------------------")
            print(ret)
            print("-------------------")

    info['gen_time'] = gentime
    gen_tokens = [len(o) - prompttoks for o in output]
    toks = sum(gen_tokens)
    if stopper:
        for seqi, stoptoks in stopper.stopped.items():
            gen_tokens[seqi] = stoptoks - prompttoks
    kepttoks = sum(gen_tokens)
    info['gen_tokens'] = gen_tokens
    print(f"<<GEN {toks} tokens ({kepttoks} useful) in {gentime :.2f}s {toks/gentime :.1f} toks/s")
    return out_text, info


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

if __name__ == '__main__':
    output, outinfo = run_llm(model2, tokenizer2, "To sum two numbers:\n```\n", 60, numseqs = 2, trim_stop = True)
