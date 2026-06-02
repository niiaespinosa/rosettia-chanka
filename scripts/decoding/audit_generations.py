"""Error-mode audit of MT generations — find the QUALITY pain points ChrF can't see.

Reads a prediction jsonl ({"source","reference","prediction"} per line) and quantifies
concrete, speaker-independent failure modes (all unambiguously bad regardless of dialect
opinion), plus dumps the worst examples for manual reading. Use this to target a
"sniper" GSPO run: penalize the modes that are frequent AND ChrF-neutral.
"""
import argparse, json, re, statistics, unicodedata
import sacrebleu

# Spanish function words that should ~never survive into good Quechua output
ES_STOP = set("el la los las un una unos unas de del al a y o u que qué porque pero "
              "con sin por para en es son era eran fue ser estar este esta esto esos "
              "esas su sus mi mis tu tus se lo le les nos me te muy más ya no si sí "
              "como cuando donde quien cual todo todos toda todas hay han ha he".split())


def words(s):
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)


def chrf(h, r):
    return sacrebleu.sentence_chrf(h, [r]).score


def adjacent_dups(ws):
    return sum(1 for i in range(1, len(ws)) if ws[i] == ws[i - 1])


def dup_bigrams(ws):
    bg = [tuple(ws[i:i + 2]) for i in range(len(ws) - 1)]
    return len(bg) - len(set(bg))


def numbers(s):
    return set(re.findall(r"\d[\d.,]*", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", required=True)
    ap.add_argument("--worst", type=int, default=25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred_jsonl) if l.strip()]
    n = len(rows)
    rep_sent = copy_sent = stop_sent = numdrop_sent = short_sent = long_sent = empty_sent = 0
    chrfs, lenratios, copyrates = [], [], []
    stop_hits = {}
    flagged = {"repetition": [], "source_copy": [], "es_stopword": [], "number_drop": [], "length": []}

    for r in rows:
        src, ref, pred = r["source"], r["reference"], r["prediction"]
        sw, rw, pw = words(src), words(ref), words(pred)
        c = chrf(pred, ref); chrfs.append(c)
        lr = (len(pw) / max(1, len(rw))); lenratios.append(lr)
        if not pw:
            empty_sent += 1
        # repetition
        ndup = adjacent_dups(pw) + dup_bigrams(pw)
        if ndup > 0:
            rep_sent += 1
            if len(flagged["repetition"]) < 12: flagged["repetition"].append((ndup, src, pred))
        # source copy: pred word-types verbatim in source (loanwords inflate this; see stopwords below)
        srcset = set(sw); shared = [w for w in set(pw) if w in srcset and not w.isdigit()]
        cr = len(shared) / max(1, len(set(pw))); copyrates.append(cr)
        if cr >= 0.30:
            copy_sent += 1
            if len(flagged["source_copy"]) < 12: flagged["source_copy"].append((round(cr, 2), src, pred))
        # Spanish stopword leakage (strong untranslated signal)
        leaked = [w for w in pw if w in ES_STOP]
        if leaked:
            stop_sent += 1
            for w in leaked: stop_hits[w] = stop_hits.get(w, 0) + 1
            if len(flagged["es_stopword"]) < 12: flagged["es_stopword"].append((leaked, src, pred))
        # number drop
        miss = numbers(src) - numbers(pred)
        if miss:
            numdrop_sent += 1
            if len(flagged["number_drop"]) < 12: flagged["number_drop"].append((sorted(miss), src, pred))
        # length anomalies
        if lr < 0.6: short_sent += 1
        if lr > 1.6: long_sent += 1
        if lr < 0.6 or lr > 1.6:
            if len(flagged["length"]) < 12: flagged["length"].append((round(lr, 2), src, pred))

    pct = lambda x: f"{x} ({100*x/n:.1f}%)"
    print(f"\n===== GENERATION AUDIT — {n} sentences ({args.pred_jsonl}) =====")
    print(f"mean sentence-ChrF = {statistics.mean(chrfs):.2f}   median = {statistics.median(chrfs):.2f}")
    print(f"mean length ratio (pred/ref words) = {statistics.mean(lenratios):.2f}")
    print(f"\n--- ERROR-MODE FREQUENCY (speaker-independent, unambiguously bad) ---")
    print(f"repetition (adjacent dup word or dup bigram) : {pct(rep_sent)}")
    print(f"Spanish stopword leakage (untranslated)      : {pct(stop_sent)}")
    print(f"high source-copy (>=30% pred types in src)   : {pct(copy_sent)}")
    print(f"number/digit dropped from source             : {pct(numdrop_sent)}")
    print(f"too short (<0.6 len ratio)                   : {pct(short_sent)}")
    print(f"too long (>1.6 len ratio)                    : {pct(long_sent)}")
    print(f"empty prediction                             : {pct(empty_sent)}")
    if stop_hits:
        top = sorted(stop_hits.items(), key=lambda x: -x[1])[:12]
        print(f"\ntop leaked Spanish words: {', '.join(f'{w}×{c}' for w,c in top)}")

    for mode, items in flagged.items():
        if not items: continue
        print(f"\n--- examples: {mode} ---")
        for tag, src, pred in items[:6]:
            print(f"  [{tag}] {src!r}\n        -> {pred!r}")

    print(f"\n--- {args.worst} WORST by sentence-ChrF (read these) ---")
    worst = sorted(rows, key=lambda r: chrf(r["prediction"], r["reference"]))[:args.worst]
    for r in worst:
        print(f"  ChrF={chrf(r['prediction'],r['reference']):.1f}")
        print(f"    SRC: {r['source']}")
        print(f"    REF: {r['reference']}")
        print(f"    HYP: {r['prediction']}")


if __name__ == "__main__":
    main()
