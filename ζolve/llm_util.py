import time
import torch
import transformers

need_safetok = True


def safe_tok_line(line, tokenizer):
    """Try to make stop token sequence robust to preceding text. Works for LLemma, Mistral. InternLM2-Math seems well behaved
    Removes the leading newline."""
    print(repr(line))
    remove_nl = False
    if not line.startswith("\n"):
        line = "\n" + line
        remove_nl = True
    nl_tok = 13  # LLEMMA and MISTRAL. 364 for InternLM2-Math
    assert tokenizer("\n", add_special_tokens = False)['input_ids'][-1] == 13
    toks = tokenizer(line, add_special_tokens = False)['input_ids']
    start = toks.index(13) + (1 if remove_nl else 0)
    print(toks, "->",toks[start:])
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
        self.are_stopped = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for seqi,sequence in enumerate(input_ids):
            for i, stop in enumerate(self.stops):
                suffix = sequence[-len(stop):]
                if torch.all(torch.eq(stop, suffix)):
                    #print("found", repr(self.stop_text[i]), self.ignore_count)
                    if self.ignore_count <= 0:
                        self.are_stopped.add( seqi)
                        break
                    print("  ignoring stop!")
                    self.ignore_count -= 1
        return len(self.are_stopped) == len(input_ids)


def run_llm(model, tokenizer, prompt, max_tokens = 800, numseqs = 1, temp = 0.9, stopwords = ["```"], show = False):
    "Returns a list of generations, not trimmed to stopwords."
    print("gen")
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = len(inputs['input_ids'][0])
    startt = time.time()
    top_p = 0.96
    #multistop_criteria.reset()

    stopper = None
    if stopwords:
        stopper = transformers.StoppingCriteriaList([MultiStoppingCriterion(model, tokenizer, stopwords)])

    output = model.generate(**inputs,
                            max_new_tokens = max_tokens,
                            pad_token_id = tokenizer.eos_token_id,
                            do_sample = True,
                            temperature = temp,
                            top_p = top_p,
                            num_return_sequences = numseqs,
                            stopping_criteria = stopper)


    #inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

    #output = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    toks = sum(len(o) - input_len for o in output)
    out_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    out_text = [text[len(prompt):] for text in out_text]
    if show:
        for text in out_text:
            print("-------------------")
            print(ret)
            print("-------------------")
    gentime =  time.time() - startt
    print(f"<<GEN {toks} tokens in {gentime :.2f}s {toks/gentime :.1f} toks/s")
    return out_text
