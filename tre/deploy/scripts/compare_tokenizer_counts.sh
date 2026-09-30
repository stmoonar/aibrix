#!/usr/bin/env bash
# Compare per-prompt token counts of two vLLM images' tokenizers (CPU only, no GPU needed).
# Evidence for the theta calibration gate of a vLLM image change (RELEASE-20260930-integration.md
# section 0 / 6.1 P3-B): a model whose counts differ must be recalibrated before a TRE-arm run.
#
# usage: compare_tokenizer_counts.sh <old image> <new image> <weights_path> <prompts.jsonl> [n=200]
#   prompts.jsonl: one JSON object per line with a "prompt" field (the calibration prompt files
#   <run>/<model>/prompts/<cell>/*.prompts.jsonl written by tre_replayer.engine.prompt_store).
# Prints one line: n, prompts that differ, max |diff|, mean old / new count, and (when the file
# carries it) the mean "prompt_tokens" the generator targeted. Exit 1 on any failure.
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,10p' "$0"; exit 2; }
OLD=$1; NEW=$2; W=$3; F=$4; N=${5:-200}
[ -s "$F" ] || { echo "empty or missing prompt file: $F" >&2; exit 1; }
[ -d "$W" ] || { echo "missing weights dir: $W" >&2; exit 1; }
OUT=$(mktemp -d /tmp/tokcount.XXXXXX); trap 'rm -rf "$OUT"' EXIT
# The in-container program, flush-left (an indented `python3 -c` body is an IndentationError).
PROG='
import json, sys
try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer
w, n = sys.argv[1], int(sys.argv[2])
t = get_tokenizer(w)
ps = []
with open("/p.jsonl") as fh:
    for line in fh:
        if len(ps) >= n:
            break
        if line.strip():
            ps.append(json.loads(line)["prompt"])
print(json.dumps({"cls": type(t).__name__,
                  "counts": [len(t.encode(p, add_special_tokens=False)) for p in ps]}))
'
for tag in old new; do
  IMG=$OLD; [ $tag = new ] && IMG=$NEW
  docker run --rm --entrypoint python3 -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 \
    -v "$W:$W:ro" -v "$F:/p.jsonl:ro" "$IMG" -c "$PROG" "$W" "$N" 2>"$OUT/$tag.err" \
    | tail -n 1 > "$OUT/$tag.json" || { cat "$OUT/$tag.err" >&2; exit 1; }
  [ -s "$OUT/$tag.json" ] || { cat "$OUT/$tag.err" >&2; exit 1; }
done
python3 - "$OUT/old.json" "$OUT/new.json" "$F" "$N" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ca, cb = a["counts"], b["counts"]
assert len(ca) == len(cb) and ca, "count lists differ in length or are empty"
d = [y - x for x, y in zip(ca, cb)]
tgt = []
with open(sys.argv[3]) as fh:
    for line in fh:
        if len(tgt) >= int(sys.argv[4]):
            break
        if line.strip():
            tgt.append(json.loads(line).get("prompt_tokens"))
tgt = [x for x in tgt if isinstance(x, (int, float))]
print(json.dumps({"n": len(d), "differ": sum(1 for x in d if x), "max_abs_diff": max(map(abs, d)),
                  "mean_old": round(sum(ca) / len(ca), 1), "mean_new": round(sum(cb) / len(cb), 1),
                  "mean_rel_diff": round(sum(y / x - 1 for x, y in zip(ca, cb) if x) / len(ca), 4),
                  "tokenizer_old": a["cls"], "tokenizer_new": b["cls"],
                  "mean_prompt_tokens_field": round(sum(tgt) / len(tgt), 1) if tgt else None}))
PY
