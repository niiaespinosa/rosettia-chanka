"""Eval an NLLB-200 (base or LoRA-fine-tuned) on AmericasNLP 2021 spa→quy test.
Reports official ChrF (word_order=0) vs v30's 40.55. Uses transformers generate
(NLLB is encoder-decoder; vLLM seq2seq support is limited)."""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="facebook/nllb-200-1.3B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
    ap.add_argument("--suppress-apostrophe", action="store_true", default=True,
                    help="forbid apostrophe generation (Ayacucho quy has no glottalization; free ChrF gain)")
    ap.add_argument("--allow-apostrophe", dest="suppress_apostrophe", action="store_false")
    ap.add_argument("--test-es", default="docs/references/americasnlp_test/2021_test.es")
    ap.add_argument("--test-quy", default="docs/references/americasnlp_test/2021_test.quy")
    ap.add_argument("--src-lang", default="spa_Latn")
    ap.add_argument("--tgt-lang", default="quy_Latn")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-pred", default=None)
    args = ap.parse_args()

    import torch, sacrebleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    src = [l.strip() for l in open(args.test_es) if l.strip()]
    ref = [l.strip() for l in open(args.test_quy) if l.strip()]
    assert len(src) == len(ref)

    tok = AutoTokenizer.from_pretrained(args.model_id, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    bos = tok.convert_tokens_to_ids(args.tgt_lang)

    # forbid apostrophe-bearing tokens (Ayacucho quy reference orthography has none)
    bad_words_ids = None
    if args.suppress_apostrophe:
        apos_chars = ["'", "’", "ʼ"]  # ' ' ʼ
        bad = set()
        vocab = tok.get_vocab()
        for t, tid in vocab.items():
            piece = t.replace("▁", "")  # strip sentencepiece marker
            if any(c in piece for c in apos_chars):
                bad.add(tid)
        bad_words_ids = [[i] for i in sorted(bad)] or None
        print(f"suppressing {len(bad)} apostrophe-bearing tokens", flush=True)

    preds = []
    for i in range(0, len(src), args.batch_size):
        batch = src[i:i+args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=bos, num_beams=args.num_beams,
                                 max_new_tokens=args.max_new, bad_words_ids=bad_words_ids)
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
        print(f"{min(i+args.batch_size,len(src))}/{len(src)}", flush=True)

    chrf0 = sacrebleu.corpus_chrf(preds, [ref], word_order=0).score
    chrfpp = sacrebleu.corpus_chrf(preds, [ref], word_order=2).score
    bleu = sacrebleu.corpus_bleu(preds, [ref]).score
    rec = {"model": args.model_id, "adapter": args.adapter,
           "ChrF_w0": chrf0, "chrf++": chrfpp, "bleu": bleu, "n": len(src)}
    json.dump(rec, open(args.out_json, "w"), indent=2)
    if args.out_pred:
        with open(args.out_pred, "w") as f:
            for s, r, p in zip(src, ref, preds):
                f.write(json.dumps({"source": s, "reference": r, "prediction": p}, ensure_ascii=False)+"\n")
    print(json.dumps(rec, indent=2))
    print(f"vs v30 40.55 | Helsinki 39.40")


if __name__ == "__main__":
    main()
