"""HEAD-TO-HEAD on identical batches: baseline a-s42 vs B'-s42.

Both checkpoints, same 24 val batches, same arithmetic. This is the only
comparison in the audit where the data, the batches, and the metric are held
fixed across the two models.

Run:  python diagnostics/audit_headtohead.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import compute_init, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

LN2 = 0.6931471805599453
B_S42 = os.path.join(get_base_dir(), "base_checkpoints", "a-s42", "model_000600.pt")
B_STAB = os.path.join(get_base_dir(), "base_checkpoints", "b-stab-s42", "model_000600.pt")


def say(s=""):
    print(s, flush=True)


def cfg_of(tok):
    V = tok.get_vocab_size()
    return GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                     n_kv_head=6, n_embd=768, window_pattern="L")


def load_rda(path, tok, device):
    base = GPT(cfg_of(tok))
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    m.eval()
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return m


def load_base(path, tok, device):
    m = GPT(cfg_of(tok))
    m.init_weights()
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return m.to(device).eval()


def bpb(model, batches, tb, rda, y_override=None):
    carry, tn, tbb = None, 0.0, 0.0
    for i, (x, y) in enumerate(batches):
        yy = y if y_override is None else y_override[i]
        with torch.no_grad():
            if rda:
                l = model(x, yy, loss_reduction="none", E_t_external=carry)
                carry = model._last_E_next_seq[:, -1, :]
            else:
                l = model(x, yy, loss_reduction="none")
        l = l.view(-1).float()
        t = yy.view(-1)
        nb = tb[torch.where(t >= 0, t, torch.zeros_like(t))]
        nb = torch.where(t >= 0, nb, torch.zeros_like(nb))
        tn += float((l * (nb > 0)).sum())
        tbb += float(nb.sum())
    return tn / tbb / LN2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_init(device, 42)
    tok = get_tokenizer()
    tb = get_token_bytes(device=device)
    ld = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="val", device=torch.device(device))
    B = []
    for _ in range(24):
        x, y, _ = next(ld)
        B.append((x.to(device), y.to(device)))

    g = torch.Generator(device="cpu").manual_seed(0)
    sh = [y[:, torch.randperm(y.size(1), generator=g).to(y.device)].contiguous()
          for _, y in B]

    base = load_base(B_S42, tok, device)
    rda = load_rda(B_STAB, tok, device)

    say("identical 24 val batches, identical arithmetic, both from ckpt\n")
    bb_t = bpb(base, B, tb, False)
    bb_s = bpb(base, B, tb, False, y_override=sh)
    say("  baseline a-s42   TRUE y      : bpb %.6f" % bb_t)
    say("  baseline a-s42   SHUFFLED y  : bpb %.6f   (delta %+.6f)" % (bb_s, bb_s - bb_t))
    say("  -> the baseline is LABEL-BLIND: shuffling y must not move it.")
    say()
    rb_t = bpb(rda, B, tb, True)
    rb_s = bpb(rda, B, tb, True, y_override=sh)
    say("  B'-s42           TRUE y      : bpb %.6f" % rb_t)
    say("  B'-s42           SHUFFLED y  : bpb %.6f   (delta %+.6f)" % (rb_s, rb_s - rb_t))
    say("  -> the B' model is NOT label-blind.")
    say()
    say("  honest comparison (both label-blind):")
    say("    baseline %.6f  vs  B' %.6f  ->  B' is %+.6f %s"
        % (bb_t, rb_s, rb_s - bb_t, "WORSE" if rb_s > bb_t else "better"))
    say()
    say("  comparison the run reported (both label-using):")
    say("    baseline %.6f  vs  B' %.6f  ->  B' is %+.6f %s"
        % (bb_t, rb_t, rb_t - bb_t, "better" if rb_t < bb_t else "worse"))


if __name__ == "__main__":
    main()
