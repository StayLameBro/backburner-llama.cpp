#!/usr/bin/env bash
# arms.sh DEPTH KV ARM... : run depth-bench arms in order, each in its OWN gpu-run.sh lock hold
# (the lock is released between arms so queued agents can run; each arm is ~1 min).
# ARM is one of: stock | R (REGFED) | RF (REGFED+FA_GQA) | plain (RF, no speculation) | ngram (RF, ngram-cache,draft-dflash)
#               | nsimple (RF, ngram-simple,draft-dflash)
# Each arm is capped at 120 s (cap.sh kills the python client and its server).
# Mirror the list for ABBA, e.g.  arms.sh 16384 f16 stock R RF RF R stock
# Extra args for depth-bench.py run go in $DB_ARGS (default "-n 128 --reps 2"; rep 0 is a warmup).
set -euo pipefail
DB="$(cd "$(dirname "$0")" && pwd)/depth-bench.py"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCK="$(cd "$HERE/../../.." && pwd)/scripts/gpu-run.sh --label depth -- $HERE/cap.sh 120"
depth=$1 kv=$2; shift 2
for arm in "$@"; do
  case $arm in
    stock) $LOCK env -u GGML_METAL_REGFED -u GGML_METAL_FA_GQA python3 "$DB" run "$depth" --kv "$kv" ${DB_ARGS:--n 128 --reps 2} ;;
    R)     $LOCK env -u GGML_METAL_FA_GQA GGML_METAL_REGFED=1 python3 "$DB" run "$depth" --kv "$kv" ${DB_ARGS:--n 128 --reps 2} ;;
    RF)    $LOCK env GGML_METAL_REGFED=1 GGML_METAL_FA_GQA=1 python3 "$DB" run "$depth" --kv "$kv" ${DB_ARGS:--n 128 --reps 2} ;;
    plain) $LOCK env GGML_METAL_REGFED=1 GGML_METAL_FA_GQA=1 python3 "$DB" run "$depth" --kv "$kv" --plain ${DB_ARGS:--n 128 --reps 2} ;;
    nsimple) $LOCK env GGML_METAL_REGFED=1 GGML_METAL_FA_GQA=1 python3 "$DB" run "$depth" --kv "$kv" --spec-type ngram-simple,draft-dflash ${DB_ARGS:--n 128 --reps 2} ;;
    ngram) $LOCK env GGML_METAL_REGFED=1 GGML_METAL_FA_GQA=1 python3 "$DB" run "$depth" --kv "$kv" --spec-type ngram-cache,draft-dflash ${DB_ARGS:--n 128 --reps 2} ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
done
