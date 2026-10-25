import copy

import numpy as np
import torch

from data.synth import finance as F
from data.synth import qa
from posttrain import methods, task
from posttrain.model import ModelConfig, TinyLM, decode, encode, generate

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=2, max_len=192)


def test_gold_trajectories_are_correct_and_valid():
    for ep in qa.make_split(300, seed=1):
        s = qa.score(ep, ep.gold_completion)
        assert s["exact"] and s["tool_ok"], (ep.prompt, ep.gold_completion)


def test_ledger_tools():
    rng = np.random.default_rng(0)
    st = F.make_statement(rng)
    led = qa.Ledger(st)
    assert led.run("C: closing()") == F.fmt_money(st.closing_cents)
    assert led.run("C: calc(10.50-0.25)") == "10.25"
    assert led.run("C: total(notacategory)") is None
    assert led.run("C: rm(-rf)") is None
    assert led.run("C: calc(1+1)") is None


def test_wrong_answer_scores_zero():
    ep = qa.make_split(1, seed=3)[0]
    assert not qa.score(ep, "C: closing()\nR: 1.00\nA: wrong\n")["exact"]
    assert not qa.score(ep, "A: " + ep.answer + "\n")["tool_ok"]  # no tool call


def test_tokenizer_roundtrip():
    s = "Q: How much was spent on dining?\nC: total(dining)\nR: 12.30\nA: 12.30\n"
    assert decode(encode(s)) == s


def test_episode_mask_excludes_prompt_and_tool_results():
    ep = qa.make_split(1, seed=4)[0]
    ids, mask = task.episode_tokens(ep)
    text = decode(ids)
    masked = "".join(c for c, m in zip(text, mask, strict=True) if m)
    assert "R: " not in masked and "Q: " not in masked and "A: " in masked


def test_kv_cache_generation_matches_full_forward():
    torch.manual_seed(0)
    m = TinyLM(TINY).eval()
    ep = qa.make_split(1, seed=5)[0]
    out = generate(m, [ep.prompt], [task.make_env(ep)], max_new=20)[0]
    ids = out["ids"]
    inp = torch.tensor([[encode("\n")[0]] + ids[:-1]])
    greedy = m(inp).argmax(-1)[0]
    for t, (g, fm) in enumerate(zip(ids, out["gen_mask"], strict=True)):
        if fm:
            assert int(greedy[t]) == g


class _Scripted(TinyLM):
    """Emits 'C: closing()' then, after the environment's R: line, 'A: <copied result>'."""

    def __init__(self, prompt):
        super().__init__(TINY)
        self.prompt, self.hist = prompt, ""

    def step(self, tok, pos, caches):
        from posttrain.model import CHARS, STOI

        self.hist += CHARS[int(tok[0])]
        after = self.hist[1 + len(self.prompt) :] if len(self.hist) > len(self.prompt) else None
        if after is None:
            nxt = "x"  # ignored: prompt positions are forced
        elif "R: " not in after:
            script = "C: closing()\n"
            nxt = script[len(after)] if len(after) < len(script) else "\n"
        elif "\n" not in after[after.index("R: ") :]:
            nxt = "x"  # R: line still being forced by the environment
        else:
            r_start = after.index("R: ")
            r_end = after.index("\n", r_start)
            script = "A: " + after[r_start + 3 : r_end] + "\n"
            rest = after[r_end + 1 :]
            nxt = script[len(rest)] if len(rest) < len(script) else "\n"
        logits = torch.full((1, len(CHARS)), -1e9)
        logits[0, STOI[nxt]] = 0.0
        return logits


def test_generation_inserts_tool_results_and_scores():
    ep = qa.make_split(1, seed=6)[0]
    ep = qa.make_episode(np.random.default_rng(0), ep.statement, qtype="closing")
    res = generate(_Scripted(ep.prompt), [ep.prompt], [task.make_env(ep)], max_new=60)[0]
    assert res["finished"]
    assert res["text"] == ep.gold_completion
    assert qa.score(ep, res["text"])["exact"]
    # R: line tokens are forced (gen_mask 0), call/answer tokens sampled (1)
    text = decode(res["ids"])
    r_pos = text.index("R: ")
    assert res["gen_mask"][r_pos] == 0 and res["gen_mask"][text.index("C: ")] == 1


def test_methods_run_and_respect_budget():
    torch.manual_seed(0)
    base = TinyLM(TINY)
    budget = 3e8
    c = methods.continued_sft(copy.deepcopy(base), 0, budget)
    assert c.flops >= budget and c.train_tokens > 0
    c, info = methods.grpo(copy.deepcopy(base), 0, budget, prompts=2, group=2)
    assert c.flops >= budget and info["iters"] >= 1
    c, info = methods.dpo(copy.deepcopy(base), 0, budget, prompts=2, k=2)
    assert c.flops >= budget and info["rounds"] >= 1


def test_reward_model_scores_in_unit_interval():
    torch.manual_seed(0)
    rm = methods.RewardModel(TinyLM(TINY))
    ep = qa.make_split(2, seed=7)
    outs = task.rollout(TinyLM(TINY), ep, None, 1.0)
    for o in outs:
        o["exact"] = False
    rm.fit(outs, task.Compute(), epochs=1, batch=2)
    s = rm.scores(outs)
    assert s.shape == (2,) and np.all((s >= 0) & (s <= 1))


def test_site_splits_are_non_iid():
    tor = qa.make_site_split(200, 0, "TOR")
    us = qa.make_site_split(200, 0, "US")
    assert all(e.statement.currency == "CAD" for e in tor) and all(e.statement.currency == "USD" for e in us)
    frac = lambda eps, ts: np.mean([e.qtype in ts for e in eps])  # noqa: E731
    assert frac(tor, {"total", "count"}) > 0.6 and frac(us, {"largest_m", "largest_a"}) > 0.6
