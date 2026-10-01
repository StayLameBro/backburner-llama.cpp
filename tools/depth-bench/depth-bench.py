#!/usr/bin/env python3
"""depth-bench: speculative-decode tok/s at a real context depth, without re-paying the prefill.

  prefill  DEPTH            prefill DEPTH tokens of real source once through llama-server (f16 KV,
                            drafter attached so its feature KV is filled too) and save the slot
                            (target state + drafter state, see tools/server slot save '.dft').
  convert  DEPTH [q8_0]     rewrite the saved f16 target KV as q8_0 (CPU only, no GPU lock needed).
  run      DEPTH            start a server, per rep: restore the slot, append the question, decode
                            N tokens greedy; print one JSON line per rep (and append to --out).

Wrap every prefill/run in the shared GPU lock:
  scripts/gpu-run.sh --label depth-bench -- python3 llama.cpp-depth/tools/depth-bench/depth-bench.py run 65536 --kv q8_0
Kernel flags are plain env vars (GGML_METAL_REGFED=1, GGML_METAL_FA_GQA=1) and are recorded.
Standard library only.
"""
import argparse, json, os, shutil, signal, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.abspath(os.path.join(HERE, '..', '..'))
ROOT = os.path.abspath(os.path.join(LLAMA, '..'))
BIN = os.environ.get('DEPTH_BENCH_BIN', os.path.join(LLAMA, 'build-depth', 'bin'))
CACHE = os.environ.get('DEPTH_BENCH_CACHE', os.path.join(ROOT, 'bench-out', 'depth-cache'))
TARGET = os.environ.get('DEPTH_BENCH_TARGET', os.path.expanduser('~/Models/') + 'Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf')
DRAFT = os.environ.get('DEPTH_BENCH_DRAFT', os.path.expanduser('~/Models/') + 'dflash2-v2-q4km.gguf')
PORT = int(os.environ.get('DEPTH_BENCH_PORT', '18089'))

# Real files a coding agent would have in context (same spirit as scripts/build-ctx-prompt.sh).
DOC_FILES = [
    'README.md', 'scripts/gpu-lock-common.sh', 'scripts/decode-bench.sh', 'docs/measurement.md',
    'llama.cpp-depth/common/speculative.cpp', 'llama.cpp-depth/src/models/dflash.cpp',
    'docs/RESULTS.md', 'llama.cpp-depth/common/arg.cpp', 'llama.cpp-depth/tools/server/server-context.cpp',
]
PREFIX = '<|im_start|>user\nRepository: infernet. The following files are the relevant source.\n\n'
QUESTION = ('\n\n===== TASK =====\n'
            'Based on the repository source above, explain step by step how the DFlash speculative '
            'implementation (common_speculative_impl_draft_dflash) injects the target model\'s layer '
            'features into the draft model\'s KV cache during prefill and during verification, and what '
            'happens to rejected draft positions. Then write a C++ function that prints, per verify round, '
            'the number of accepted tokens and the time spent drafting versus verifying.'
            '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')

FLAG_ENV = ['GGML_METAL_REGFED', 'GGML_METAL_FA_GQA', 'GGML_METAL_FA_GQA_NWG', 'SPEC_DFLASH_TIMING']


def http(method, path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')


class Server:
    def __init__(self, depth, kv, spec, log, extra=(), spec_type='draft-dflash'):
        n_ctx = depth + 1536
        self.cmd = [os.path.join(BIN, 'llama-server'), '-m', TARGET, '-ngl', '999', '-fa', 'on',
                    '-c', str(n_ctx), '-np', '1', '--cache-ram', '0', '--slot-save-path', CACHE + '/',
                    '-ctk', kv, '-ctv', kv, '--port', str(PORT), '--host', '127.0.0.1', '--no-webui']
        if not os.environ.get('DEPTH_BENCH_NO_BS'):
            self.cmd.append('-bs')  # backend (GPU) sampling
        if spec:
            self.cmd += ['--spec-type', spec_type, '-md', DRAFT, '-ngld', '999',
                         '--spec-draft-n-max', str(spec)]
        self.cmd += list(extra) + os.environ.get('DEPTH_BENCH_EXTRA', '').split()  # extra server args
        self.log = log

    def __enter__(self):
        self.logf = open(self.log, 'w')
        self.logf.write(' '.join(self.cmd) + '\n'); self.logf.flush()
        self.p = subprocess.Popen(self.cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise SystemExit(f'server exited ({self.p.returncode}); see {self.log}')
            try:
                st, _ = http('GET', '/health', timeout=5)
                if st == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > 600:
                raise SystemExit('server did not come up in 600 s')
            time.sleep(1)
        self.t_load = time.time() - t0
        return self

    def __exit__(self, *a):
        self.p.send_signal(signal.SIGINT)
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.logf.close()


def tokenize(text):
    st, r = http('POST', '/tokenize', {'content': text, 'add_special': False, 'parse_special': True})
    assert st == 200, r
    return r['tokens']


def build_doc():
    parts = []
    for f in DOC_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            parts.append(f'===== {f} =====\n' + open(p, errors='replace').read() + '\n\n')
    return ''.join(parts)


def state_name(depth, kv):
    return f'd{depth}-{kv}.bin'


def cmd_prefill(a):
    os.makedirs(CACHE, exist_ok=True)
    log = os.path.join(CACHE, f'prefill-d{a.key}-{a.kv}.log' if a.kv != 'f16' else f'prefill-d{a.key}.log')
    if a.kv != 'f16':
        # native prefill with a quantized KV of the SAME tokens as the f16 cache (validity gate for convert)
        meta = json.load(open(os.path.join(CACHE, f'd{a.key}.json')))
        name = f'd{a.key}-{a.kv}-native.bin'
        with Server(a.depth, a.kv, a.n_draft, log) as srv:
            t0 = time.time()
            st, r = http('POST', '/completion', {'prompt': meta['prefix'], 'n_predict': 1, 'id_slot': 0,
                                                 'cache_prompt': True, 'temperature': 0})
            assert st == 200, r
            t_prefill = time.time() - t0
            st, s = http('POST', '/slots/0?action=save', {'filename': name})
            assert st == 200, s
        print(json.dumps({'depth': a.depth, 'kv': a.kv, 'state': name, 'prefill_s': round(t_prefill, 1), 'save': s}))
        return
    with Server(a.depth, 'f16', a.n_draft, log) as srv:
        if a.doc:
            # a prepared long-context prompt whose task follows a '===== TASK =====' marker
            # (e.g. bench-out/ctx60k.txt): prefill the document part, keep the task as the suffix
            text = open(a.doc, errors='replace').read()
            body, sep, task = text.rpartition('\n\n===== TASK =====\n')
            if not sep:
                raise SystemExit(f'{a.doc}: no "===== TASK =====" marker')
            head = tokenize('<|im_start|>user\n')
            doc = tokenize(body)
            question = sep + task.rstrip('\n') + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            if len(head) + len(doc) < a.depth:
                print(f'NOTE: document is {len(head) + len(doc)} tokens < {a.depth}; prefilling all of it',
                      file=sys.stderr)
        else:
            head = tokenize(PREFIX)
            doc = tokenize(build_doc())
            question = QUESTION
            if len(head) + len(doc) < a.depth:
                raise SystemExit(f'document too short: {len(head) + len(doc)} < {a.depth} tokens')
        prefix = head + doc[:a.depth - len(head)]
        suffix = tokenize(question)
        t0 = time.time()
        st, r = http('POST', '/completion', {'prompt': prefix, 'n_predict': 1, 'id_slot': 0,
                                             'cache_prompt': True, 'temperature': 0})
        t_prefill = time.time() - t0
        assert st == 200, r
        st, s = http('POST', '/slots/0?action=save', {'filename': state_name(a.key, 'f16')})
        assert st == 200, s
        meta = {'depth': a.depth, 'prefix': prefix, 'suffix': suffix, 'prefill_s': round(t_prefill, 1),
                'prefill_timings': r.get('timings'), 'save': s, 'target': TARGET, 'draft': DRAFT,
                'doc': a.doc, 'n_prefix': len(prefix)}
        json.dump(meta, open(os.path.join(CACHE, f'd{a.key}.json'), 'w'))
        print(json.dumps({k: meta[k] for k in ('depth', 'n_prefix', 'prefill_s', 'save')}))
        if s.get('n_saved') not in (None, len(prefix)):
            print(f'WARNING: saved {s.get("n_saved")} tokens, prefix has {len(prefix)}', file=sys.stderr)


def cmd_convert(a):
    src = os.path.join(CACHE, state_name(a.key, 'f16'))
    dst = os.path.join(CACHE, state_name(a.key, a.kv))
    subprocess.check_call([os.path.join(BIN, 'llama-kvstate-convert'), src, dst, a.kv])
    # the drafter state does not depend on the target KV type
    if os.path.exists(src + '.dft') and not os.path.exists(dst + '.dft'):
        os.symlink(os.path.basename(src) + '.dft', dst + '.dft')


def cmd_run(a):
    meta = json.load(open(os.path.join(CACHE, f'd{a.key}.json')))
    fname = a.state or state_name(a.key, a.kv)
    if not os.path.exists(os.path.join(CACHE, fname)):
        raise SystemExit(f'missing {fname}: run prefill (and convert for q8_0) first')
    flags = {k: os.environ[k] for k in FLAG_ENV if k in os.environ}
    label = a.label or ('+'.join(k.replace('GGML_METAL_', '') for k in flags if flags[k] not in ('', '0')
                                 and k != 'SPEC_DFLASH_TIMING') or 'stock')
    mode = 'plain' if a.plain else (f'dflash{a.n_draft}' if a.spec_type == 'draft-dflash'
                                     else f'{a.spec_type.replace(",", "+")}{a.n_draft}')
    task = a.task or 'orig'
    if a.state:
        label += '-' + os.path.splitext(a.state)[0]
    log = os.path.join(CACHE, f'run-d{a.key}-{a.kv}-{mode}-{label}-{task}.log')
    rows = []
    with Server(a.depth, a.kv, 0 if a.plain else a.n_draft, log, spec_type=a.spec_type) as srv:
        if a.task:
            # a new task on the cached document: same marker and chat tail as the prefilled one
            text = open(os.path.join(HERE, 'tasks', a.task + '.txt')).read().strip()
            think = '<think>\n' if a.think else '<think>\n\n</think>\n\n'   # --think: the model reasons first, as omp runs it
            suffix = tokenize('\n\n===== TASK =====\n' + text + '<|im_end|>\n<|im_start|>assistant\n' + think)
        else:
            suffix = meta['suffix']
        prompt = meta['prefix'] + suffix
        for rep in range(a.reps):
            st, rs = http('POST', '/slots/0?action=restore', {'filename': fname})
            assert st == 200, rs
            st, r = http('POST', '/completion', {'prompt': prompt, 'n_predict': a.n, 'id_slot': 0,
                                                 'cache_prompt': True, 'temperature': a.temp, 'ignore_eos': True,
                                                 **({'top_k': a.top_k, 'top_p': a.top_p, 'min_p': a.min_p, 'seed': a.seed + rep} if a.temp > 0 else {}),
                                                 **({'samplers': a.samplers.split(',')} if a.samplers else {})})
            assert st == 200, r
            t = r['timings']
            row = {'depth': a.depth, 'kv': a.kv, 'mode': mode, 'flags': label, 'task': task, 'tag': a.tag, 'rep': rep,
                   'temp': a.temp, 'think': a.think,
                   'cache_n': t.get('cache_n'), 'prompt_n': t.get('prompt_n'),
                   'n_gen': t['predicted_n'], 'tok_s': round(t['predicted_per_second'], 2),
                   'restore_ms': round(rs.get('timings', {}).get('restore_ms', 0), 0),
                   'suffix_ms': round(t.get('prompt_ms', 0), 0)}  # suffix prefill: ~2.4 s at 60k; far more = GPU/memory contention
            nv = t.get('n_tgt_decode', 0)
            if t.get('draft_verif_steps'):
                row['acc_len'] = round(1 + t['draft_n_accepted'] / t['draft_verif_steps'], 2)
                row['draft_accept'] = round(t['draft_n_accepted'] / max(1, t['draft_n']), 3)
            if nv:
                row['verify_ms'] = round(t['t_tgt_decode_ms'] / nv, 2)
                nd = max(1, t.get('n_spec_draft', 0))
                if t.get('n_spec_draft'):
                    row['draft_ms'] = round(t['t_spec_draft_ms'] / nd, 2)
                    row['inject_ms'] = round(t['t_spec_process_ms'] / nd, 2)
                row['blocks'] = nv
                row['other_ms'] = round((t['predicted_ms'] - t['t_tgt_decode_ms'] - t.get('t_spec_draft_ms', 0)
                                         - t.get('t_spec_process_ms', 0)) / nv, 2)
            if rep == 0 and a.reps > 1:
                row['warmup'] = True
            if t.get('spec_impls'):
                row['impls'] = {i['type']: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in i.items()
                                            if k != 'type'} for i in t['spec_impls']}
            row['text_head'] = r.get('content', '')[:60]
            if a.full_text:
                row['text'] = r.get('content', '')
            rows.append(row)
            print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, 'a') as f:
            for row in rows:
                row['load_s'] = round(srv.t_load, 1)
                row['date'] = time.strftime('%Y-%m-%d %H:%M')
                f.write(json.dumps(row) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('prefill'); p.add_argument('depth', type=int)
    p.add_argument('--n-draft', type=int, default=7)
    p.add_argument('--doc', help='prepared prompt file with a "===== TASK =====" tail (e.g. bench-out/ctx60k.txt)')
    p.add_argument('--kv', default='f16', help='non-f16: native prefill of the saved d{DEPTH}.json prefix, saved as d{DEPTH}-{kv}-native.bin')
    p = sp.add_parser('convert'); p.add_argument('depth', type=int); p.add_argument('kv', nargs='?', default='q8_0')
    p = sp.add_parser('run'); p.add_argument('depth', type=int)
    p.add_argument('--kv', default='f16'); p.add_argument('--plain', action='store_true')
    p.add_argument('--n-draft', type=int, default=7); p.add_argument('-n', type=int, default=256)
    p.add_argument('--spec-type', default='draft-dflash', help='e.g. ngram-cache,draft-dflash')
    p.add_argument('--reps', type=int, default=2, help='rep 0 is a warmup (shader compile, page-in)')
    p.add_argument('--state', help='state file name in the cache (default d{DEPTH}-{kv}.bin)')
    p.add_argument('--task', help='task name: tools/depth-bench/tasks/NAME.txt replaces the prefilled task')
    p.add_argument('--samplers', help='comma list for the request, e.g. "temperature" = greedy without the default top_k/top_p/min_p chain')
    p.add_argument('--full-text', action='store_true', help='store the whole generated text in the row')
    p.add_argument('--temp', type=float, default=0.0, help='0 = greedy (the old default); the GGUF recommends 1.0 with top_k 20 / top_p 0.95 / min_p 0.05')
    p.add_argument('--top-k', type=int, default=20); p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--min-p', type=float, default=0.05); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--think', action='store_true', help='open a <think> block (task only) instead of the empty one')
    p.add_argument('--label'); p.add_argument('--out', default=os.path.join(ROOT, 'bench-out', 'depth-bench.jsonl'))
    for q in sp.choices.values():
        q.add_argument('--tag', default='', help='cache key suffix, e.g. "doc" for a --doc prefill at a depth that already has a cache')
    a = ap.parse_args()
    a.key = f'{a.depth}{a.tag}'
    {'prefill': cmd_prefill, 'convert': cmd_convert, 'run': cmd_run}[a.cmd](a)


if __name__ == '__main__':
    main()
