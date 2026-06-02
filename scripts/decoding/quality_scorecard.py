"""Beyond-the-metric QUALITY SCORECARD for ultra-low-resource MT.

ChrF saturates and (single-reference) hides quality differences behind paraphrase
variance. This scorecard scores MT outputs on several speaker-free axes so we can
compare systems by *quality*, not just one surface number:

  1. ChrF (w0)            — surface fidelity vs reference (the standard metric)
  2. Adequacy (round-trip)— reverse-translate pred quy->spa, ChrF vs the ORIGINAL
                            Spanish source. Comparison is Spanish-Spanish => reliable;
                            measures meaning preservation WITHOUT needing a quy metric.
  3. Degeneracy rate      — % sentences with a true repetition loop (NOT grammatical
                            Quechua reduplication: we require >=4 repeats / char-runs)
  4. Untranslated leakage — % sentences with a Spanish function word surviving
  5. Number convention    — digit-retention rate on number-bearing sources
  6. Length calibration   — mean |pred/ref length ratio - 1|

Run with >=2 systems' prediction dumps to get a comparison table. Adequacy needs a
reverse quy->spa model (--reverse-adapter); omit --reverse-adapter to skip that axis.

This reframes the contribution as "quality beyond the metric" — honest about ChrF's
ceiling, and a reusable evaluation protocol for low-resource MT with no speakers.
"""
import argparse, json, re, statistics
import sacrebleu

ES_STOP = set("el la los las un una unos unas de del al y o que porque pero con sin por "
              "para en es son era eran fue este esta esto su sus mi tu se lo le les nos me "
              "te mas ya no si como cuando donde quien cual todo todos hay han ha".split())


def words(s): return re.findall(r"\w+", s.lower(), flags=re.UNICODE)
def chrf(h, r): return sacrebleu.sentence_chrf(h, [r]).score
def has_digit(s): return bool(re.search(r"\d", s))


def has_loop(pred):
    """True degenerate repetition, distinct from grammatical reduplication (<=2-3x)."""
    ws = pred.split()
    run = 1
    for i in range(1, len(ws)):
        run = run + 1 if ws[i] == ws[i - 1] else 1
        if run >= 4:                      # same word 4+ times in a row
            return True
    for w in ws:                          # char-run inside one token, e.g. 'llipipipi...'
        if len(w) >= 20:
            for k in (1, 2, 3):
                if len(w) >= 4 * k and w[:k] * 4 in w:
                    return True
    return False


def load_preds(spec):
    label, path = spec.split("=", 1) if "=" in spec else (spec, spec)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return label, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", nargs="+", required=True,
                    help="one or more 'label=path.jsonl' ({source,reference,prediction})")
    ap.add_argument("--base", default="facebook/nllb-200-1.3B")
    ap.add_argument("--reverse-adapter", default=None, help="quy->spa model for adequacy; omit to skip")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    systems = [load_preds(s) for s in args.pred_jsonl]

    backs = {}
    if args.reverse_adapter:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(args.base, src_lang="quy_Latn", tgt_lang="spa_Latn")
        m = AutoModelForSeq2SeqLM.from_pretrained(args.base, dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(m, args.reverse_adapter).to("cuda").eval()
        fbos = tok.convert_tokens_to_ids("spa_Latn")

        @torch.no_grad()
        def rev(texts, bs=96):
            out = []
            for i in range(0, len(texts), bs):
                enc = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=160).to("cuda")
                g = m.generate(**enc, forced_bos_token_id=fbos, num_beams=1, max_new_tokens=160)
                out.extend(tok.batch_decode(g, skip_special_tokens=True))
            return out
        for label, rows in systems:
            backs[label] = rev([r["prediction"] for r in rows])

    report = {}
    for label, rows in systems:
        n = len(rows)
        cf = sacrebleu.corpus_chrf([r["prediction"] for r in rows], [[r["reference"] for r in rows]], word_order=0).score
        loops = sum(has_loop(r["prediction"]) for r in rows)
        leak = sum(any(w in ES_STOP for w in words(r["prediction"])) for r in rows)
        lr = [len(words(r["prediction"])) / max(1, len(words(r["reference"]))) for r in rows]
        numsrc = [r for r in rows if has_digit(r["source"])]
        digit_keep = sum(has_digit(r["prediction"]) for r in numsrc) / max(1, len(numsrc))
        adeq = None
        if label in backs:
            adeq = statistics.mean(chrf(b, r["source"]) for b, r in zip(backs[label], rows))
        report[label] = {
            "ChrF_w0": round(cf, 2),
            "adequacy_roundtrip": round(adeq, 2) if adeq is not None else None,
            "degeneracy_loop_%": round(100 * loops / n, 2),
            "es_leakage_%": round(100 * leak / n, 2),
            "number_digit_retention_%": round(100 * digit_keep, 1),
            "len_miscalibration": round(statistics.mean(abs(x - 1) for x in lr), 3),
            "n": n,
        }

    axes = ["ChrF_w0", "adequacy_roundtrip", "degeneracy_loop_%", "es_leakage_%",
            "number_digit_retention_%", "len_miscalibration"]
    print("\n===== MT QUALITY SCORECARD =====")
    w = max(len(l) for l, _ in systems)
    print(f"{'system':<{w}} " + " ".join(f"{a:>22}" for a in axes))
    for label, _ in systems:
        r = report[label]
        print(f"{label:<{w}} " + " ".join(f"{str(r[a]):>22}" for a in axes))
    print("\n(higher better: ChrF, adequacy. lower better: degeneracy, leakage, len_miscalibration.")
    print(" number_digit_retention is descriptive — refs split ~48/52 digit/spelled-out.)")
    if args.out_json:
        json.dump(report, open(args.out_json, "w"), indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
