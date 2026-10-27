#!/usr/bin/env python3
"""Local, deadline-aware vLLM chat/SSE benchmark. Python standard library only."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time
import urllib.error
import urllib.request
import uuid


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def quantile(values, q):
    values = sorted(x for x in values if x is not None and math.isfinite(x))
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


class Client:
    def __init__(self, args, stop_at):
        self.args, self.stop_at = args, stop_at
        self.base = args.base_url.rstrip('/')
        self.root = self.base[:-3] if self.base.endswith('/v1') else self.base
        if not self.base.endswith('/v1'):
            self.base += '/v1'
        self.headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        key = os.environ.get('VLLM_BENCH_API_KEY')
        if key:
            self.headers['Authorization'] = 'Bearer ' + key

    def remaining(self):
        return self.stop_at - time.monotonic()

    def call(self, url, body=None, timeout=None):
        left = self.remaining()
        if left <= 0:
            raise TimeoutError('Global benchmark deadline reached')
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=self.headers)
        return urllib.request.urlopen(req, timeout=max(.05, min(timeout or 60, left)))

    def json_call(self, path, body=None):
        with self.call(self.root + path, body) as response:
            return json.load(response)

    def tokenize(self, messages=None, text=None):
        body = {'model': self.args.model, 'add_special_tokens': False}
        if messages is not None:
            body.update(messages=messages, add_generation_prompt=True,
                        chat_template_kwargs={'thinking': False})
        else:
            body['prompt'] = text
        result = self.json_call('/tokenize', body)
        return int(result.get('count', len(result.get('tokens', []))))


WORDS = ('river forest mountain valley garden ocean cloud meadow bridge island '
         'copper silver amber willow cedar maple stone wheat orchard harbor '
         'window table book lamp road basket canvas paper morning evening').split()


def make_prompt(client, target, nonce):
    """Calibrate rendered chat length, retaining a unique nonce at the very start."""
    rng = random.Random(nonce)
    words = [rng.choice(WORDS) for _ in range(target * 3)]
    beginning = (f'{nonce}\nWrite a detailed, coherent explanation of how a community '
                 'can design and maintain a public garden. Use the following unrelated '
                 'reference words only as background; do not quote or list them. '
                 'Continue with useful detail until the response limit.\nReference: ')
    lo, hi = 0, len(words)
    best = None
    for _ in range(16):
        n = (lo + hi) // 2
        content = beginning + ' '.join(words[:n]) + '\nNow provide the explanation.'
        messages = [{'role': 'user', 'content': content}]
        count = client.tokenize(messages=messages)
        candidate = {'messages': messages, 'prompt_tokens_tokenizer': count,
                     'target_prompt_tokens': target, 'unique_prefix': nonce,
                     'prompt_sha256': hashlib.sha256(content.encode()).hexdigest()}
        if best is None or abs(count - target) < abs(best['prompt_tokens_tokenizer'] - target):
            best = candidate
        if abs(count - target) <= client.args.prompt_tolerance:
            return candidate
        if count < target:
            lo = n + 1
        else:
            hi = n - 1
        if lo > hi:
            break
    if abs(best['prompt_tokens_tokenizer'] - target) > client.args.prompt_tolerance:
        raise ValueError(f'Could not calibrate {target}-token prompt: best {best["prompt_tokens_tokenizer"]}')
    return best


def timing_metrics(record):
    events = record['token_events']
    n = record.get('completion_tokens')
    first = events[0]['t_s'] if events else None
    last = events[-1]['t_s'] if events else None
    total_observed = sum(e['tokens'] for e in events if e['tokens'] is not None)
    reliable_events = bool(events) and all(e['tokens'] is not None for e in events)
    exact_itl = reliable_events and total_observed == n and all(e['tokens'] == 1 for e in events)
    gaps = [(events[i]['t_s'] - events[i - 1]['t_s']) * 1000 for i in range(1, len(events))]
    amortized = []
    if reliable_events:
        for i, gap in enumerate(gaps, 1):
            amortized += [gap / events[i]['tokens']] * events[i]['tokens']
    record.update(
        ttft_s=first, first_visible_text_s=record.get('first_visible_text_s'),
        last_token_s=last, observed_output_tokens=total_observed if reliable_events else None,
        token_arrival_count_complete=reliable_events and total_observed == n,
        token_event_count=len(events), coalesced_token_events=sum(e['tokens'] is not None and e['tokens'] > 1 for e in events),
        itl_exact_at_client=exact_itl,
        itl_p50_ms=quantile(gaps, .5) if exact_itl else None,
        itl_p95_ms=quantile(gaps, .95) if exact_itl else None,
        stream_gap_p50_ms=quantile(gaps, .5), stream_gap_p95_ms=quantile(gaps, .95),
        amortized_itl_p50_ms=quantile(amortized, .5), amortized_itl_p95_ms=quantile(amortized, .95),
        decode_tps=None, e2e_output_tps=None)
    if n is not None and record.get('elapsed_s', 0) > 0:
        record['e2e_output_tps'] = n / record['elapsed_s']
    if reliable_events and total_observed == n and last is not None and last > first:
        record['decode_tps'] = (n - events[0]['tokens']) / (last - first)
    record['ttft_source'] = ('token_counted_SSE_arrival' if reliable_events else 'first_nonempty_text_SSE_arrival')
    record['decode_tps_definition'] = '(observed generated tokens after first event)/(last - first token event)'
    return record


def run_request(client, prompt, request_id, barrier, condition):
    args = client.args
    body = {'model': args.model, 'messages': prompt['messages'], 'temperature': 0,
            'max_tokens': args.output_tokens, 'stream': True,
            'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
            'stream_interval': 1, 'chat_template_kwargs': {'thinking': False},
            'ignore_eos': not args.allow_early_eos}
    if args.return_token_ids:
        body['return_token_ids'] = True
    record = dict(request_id=request_id, condition=condition, prompt=prompt,
                  request_body=body, raw_sse=[], token_events=[], usage=None,
                  output_text='', reasoning_text='', success=False, warnings=[])
    barrier.wait()
    start = time.monotonic()
    record.update(started_utc=utc(), start_monotonic=start)
    request_deadline = min(client.stop_at, start + args.request_timeout)
    observed = 0
    stream_done = False
    try:
        with client.call(client.base + '/chat/completions', body,
                         timeout=min(args.request_timeout, max(.05, client.remaining()))) as response:
            record['http_status'] = response.status
            record['response_headers'] = {k: v for k, v in response.headers.items()
                                          if k.lower() in ('content-type', 'server', 'x-request-id')}
            event_lines = []
            while True:
                left = request_deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('Request/global deadline reached while streaming')
                # Bound each blocking read by the actual remaining deadline.
                try:
                    response.fp.raw._sock.settimeout(left)
                except AttributeError:
                    pass
                line = response.readline()
                if not line:
                    break
                decoded = line.decode('utf-8').rstrip('\r\n')
                if decoded:
                    if decoded.startswith('data:'):
                        event_lines.append(decoded[5:].lstrip())
                    continue
                if not event_lines:
                    continue
                payload = '\n'.join(event_lines)
                event_lines = []
                elapsed = time.monotonic() - start
                if payload == '[DONE]':
                    record['raw_sse'].append({'t_s': elapsed, 'data': '[DONE]'})
                    stream_done = True
                    break
                obj = json.loads(payload)
                record['raw_sse'].append({'t_s': elapsed, 'data': obj})
                if obj.get('error'):
                    raise RuntimeError(json.dumps(obj['error']))
                if obj.get('usage') is not None:
                    record['usage'] = obj['usage']
                choices = obj.get('choices') or []
                if not choices:
                    continue  # A final usage-only event is not a token arrival.
                choice = choices[0]
                delta = choice.get('delta') or {}
                text = delta.get('content') or ''
                reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
                if text and record.get('first_visible_text_s') is None:
                    record['first_visible_text_s'] = elapsed
                record['output_text'] += text
                record['reasoning_text'] += reasoning
                if choice.get('finish_reason') is not None:
                    record['finish_reason'] = choice['finish_reason']
                ids = choice.get('token_ids')
                cumulative = (obj.get('usage') or {}).get('completion_tokens')
                increment, source = None, None
                if ids is not None:
                    increment, source = len(ids), 'stream_token_ids'
                elif cumulative is not None:
                    increment, source = max(0, int(cumulative) - observed), 'continuous_usage'
                if increment is not None:
                    if increment > 0:
                        record['token_events'].append({'t_s': elapsed, 'tokens': increment, 'source': source})
                        observed += increment
                elif text or reasoning:
                    record['token_events'].append({'t_s': elapsed, 'tokens': None, 'source': 'text_chunk'})
            record['elapsed_s'] = time.monotonic() - start
        record['stream_done'] = stream_done
        if not stream_done:
            raise RuntimeError('Stream ended without [DONE]')
        usage = record.get('usage') or {}
        record['completion_tokens'] = usage.get('completion_tokens')
        record['prompt_tokens'] = usage.get('prompt_tokens', prompt['prompt_tokens_tokenizer'])
        record['completion_count_source'] = 'server_usage' if record['completion_tokens'] is not None else None
        if record['completion_tokens'] is None and observed > 0:
            record['completion_tokens'] = observed
            record['completion_count_source'] = 'stream_token_ids_or_continuous_usage'
        record['success'] = bool(record.get('finish_reason'))
        if not record['success']:
            record['error'] = 'No finish_reason in completed stream'
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, urllib.error.HTTPError):
            record['http_status'] = exc.code
            try:
                record['error_body'] = exc.read(65536).decode('utf-8', errors='replace')
            except Exception:
                pass
            finally:
                exc.close()
    record['end_monotonic'] = time.monotonic()
    record.setdefault('elapsed_s', record['end_monotonic'] - start)
    # Fallback text tokenization is deliberately OUTSIDE the timed request window.
    if record.get('success') and record.get('completion_tokens') is None:
        try:
            record['completion_tokens'] = client.tokenize(text=record['reasoning_text'] + record['output_text'])
            record['completion_count_source'] = 'retokenized_text_excludes_hidden_or_special_tokens'
            record['warnings'].append('Retokenized text is an approximation to actual generation token count')
        except Exception as exc:
            record['warnings'].append(f'Output token count unavailable: {exc}')
    record['meets_output_target'] = record.get('completion_tokens') == args.output_tokens
    if record['reasoning_text']:
        record['warnings'].append('Reasoning text observed despite thinking=false')
    if record.get('prompt_tokens') != prompt['prompt_tokens_tokenizer']:
        record['warnings'].append('Completion usage prompt count differs from calibration')
    return timing_metrics(record)


def run_wave(client, prompts, condition, records_path):
    barrier = threading.Barrier(len(prompts) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(run_request, client, p, uuid.uuid4().hex, barrier, condition) for p in prompts]
        barrier.wait()
        records = []
        with records_path.open('a') as fp:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                fp.write(json.dumps(result) + '\n')
                fp.flush()
                records.append(result)
    wall = max(r['end_monotonic'] for r in records) - min(r['start_monotonic'] for r in records)
    successes = [r for r in records if r['success']]
    counts_known = all(r.get('completion_tokens') is not None for r in successes)
    completed_tokens = sum(r.get('completion_tokens') or 0 for r in successes)
    def percentile(field, q):
        return quantile([r.get(field) for r in successes], q)
    gaps = []
    for r in successes:
        if r['itl_exact_at_client']:
            ev = r['token_events']
            gaps += [(ev[i]['t_s'] - ev[i - 1]['t_s']) * 1000 for i in range(1, len(ev))]
    all_exact = len(successes) == len(records) and all(r['itl_exact_at_client'] for r in successes)
    return dict(condition=condition, requests=len(records), successes=len(successes),
                failures=len(records) - len(successes), wall_s=wall,
                generated_tokens_successful=completed_tokens,
                aggregate_output_tps=completed_tokens / wall if counts_known and len(successes) == len(records) else None,
                partial_success_output_tps=completed_tokens / wall if counts_known else None,
                ttft_p50_s=percentile('ttft_s', .5), ttft_p95_s=percentile('ttft_s', .95),
                user_decode_tps_p50=percentile('decode_tps', .5),
                user_decode_tps_min=min((r['decode_tps'] for r in successes if r['decode_tps'] is not None), default=None),
                itl_p50_ms=quantile(gaps, .5) if all_exact else None,
                itl_p95_ms=quantile(gaps, .95) if all_exact else None,
                itl_all_requests_exact=all_exact,
                output_target_matches=sum(r['meets_output_target'] for r in records),
                actual_prompt_tokens=[r.get('prompt_tokens', r['prompt']['prompt_tokens_tokenizer']) for r in records],
                cached_prompt_tokens=[(r.get('usage') or {}).get('prompt_tokens_details', {}).get('cached_tokens')
                                      if isinstance((r.get('usage') or {}).get('prompt_tokens_details'), dict) else None for r in records],
                request_ids=[r['request_id'] for r in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='dsv41-1017-heretic')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--deadline-utc', help='Optional ISO-8601 UTC hard stop; earlier deadline wins')
    parser.add_argument('--request-timeout', type=float, default=180)
    parser.add_argument('--min-wave-budget', type=float, default=30)
    parser.add_argument('--prompt-tokens', nargs='+', type=int, default=[2048, 8192])
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--unique-prefix-only', action='store_true',
                        help='Skip repeated warm waves to prioritize concurrency coverage')
    parser.add_argument('--prompt-tolerance', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--allow-early-eos', action='store_true')
    parser.add_argument('--return-token-ids', action='store_true', help='Additional exact IDs; also sends full prompt IDs in initial SSE')
    parser.add_argument('--conditions-json', type=Path, help='Hardware, engine/weights revision, launch flags and topology to copy into manifest')
    parser.add_argument('--label', default='full-model trial')
    args = parser.parse_args()
    cache_conditions = ('unique_prefix',) if args.unique_prefix_only else ('unique_prefix', 'repeat_requested_warm')
    if any(x < 1 for x in args.concurrency + args.prompt_tokens) or args.repeats < 1:
        parser.error('Concurrency, prompt lengths and repeats must be positive')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be a new or empty directory to avoid mixing runs')
    args.out.mkdir(parents=True, exist_ok=True)
    stop_at = time.monotonic() + args.max_seconds
    if args.deadline_utc:
        end = dt.datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
        if end.tzinfo is None:
            parser.error('--deadline-utc must include Z or UTC offset')
        stop_at = min(stop_at, time.monotonic() + (end - dt.datetime.now(dt.timezone.utc)).total_seconds())
    client = Client(args, stop_at)
    manifest = dict(started_utc=utc(), label=args.label,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python_version=__import__('sys').version, warmup_generation_requests=0,
                    measurements='Client observed over supplied endpoint/tunnel; network and scheduler queueing included',
                    prompt_method='Unique first-content nonce and randomized words; full chat count calibrated by /tokenize',
                    cache_conditions=list(cache_conditions),
                    cache_method=('Unique-prefix waves only; repeated warm conditions were not measured'
                                  if args.unique_prefix_only else
                                  'Repeat the exact cold-wave prompts immediately; warm is requested, not guaranteed; use cached_tokens when available'),
                    percentile_method='Linear interpolation; small request counts are descriptive, not stable tail estimates')
    if args.conditions_json:
        manifest['server_conditions'] = json.loads(args.conditions_json.read_text())
    waves, skipped = [], []
    manifest_path = args.out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    error = None
    try:
        manifest['models_response'] = client.json_call('/v1/models')
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        # C1/2/3/4 in sequence, alternating 2K/8K per C for useful partial coverage.
        for concurrency in args.concurrency:
            for target in args.prompt_tokens:
                for repetition in range(args.repeats):
                    key = dict(concurrency=concurrency, target_prompt_tokens=target, repetition=repetition)
                    if client.remaining() < args.min_wave_budget:
                        skipped.append(dict(**key, reason='insufficient remaining budget for preparation/new wave'))
                        continue
                    prompts = [make_prompt(client, target, uuid.uuid4().hex) for _ in range(concurrency)]
                    for cache in cache_conditions:
                        condition = dict(**key, cache=cache)
                        if client.remaining() < args.min_wave_budget:
                            skipped.append(dict(**condition, reason='insufficient remaining budget'))
                            continue
                        wave = run_wave(client, prompts, condition, args.out / 'requests.jsonl')
                        waves.append(wave)
                        print(json.dumps(wave), flush=True)
                        (args.out / 'summary.json').write_text(json.dumps(dict(waves=waves, skipped=skipped), indent=2))
                        if wave['successes'] == 0:
                            raise RuntimeError('Every request in a wave failed; stopping to avoid spending the remaining budget on the same error')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, flush=True)
    manifest.update(finished_utc=utc(), error=error)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (args.out / 'summary.json').write_text(json.dumps(dict(waves=waves, skipped=skipped, error=error), indent=2))
    lines = ['# Streaming trial measurements', '',
             '| Prompt target | C | Cache condition | Success/total | TTFT p50 s | User decode p50 tok/s | Aggregate output tok/s | ITL p50 / p95 ms |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'unavailable' if x is None else f'{x:.3f}'
    for w in waves:
        c = w['condition']
        lines.append(f"| {c['target_prompt_tokens']} | {c['concurrency']} | {c['cache']} | {w['successes']}/{w['requests']} | {fmt(w['ttft_p50_s'])} | {fmt(w['user_decode_tps_p50'])} | {fmt(w['aggregate_output_tps'])} | {fmt(w['itl_p50_ms'])} / {fmt(w['itl_p95_ms'])} |")
    lines += ['', 'Aggregate includes prefill and the complete concurrent wave. Decode excludes time before the first token.',
              'ITL is shown only when every generated token has an individual observed SSE arrival. Coalesced alternatives are in raw records.',
              ('Only unique-prefix waves were requested; repeated warm conditions were not measured.'
               if args.unique_prefix_only else
               'Warm means an exact prompt repeat was requested; verify cached token counts before claiming a cache hit.'),
              f'Skipped conditions: {len(skipped)}. Terminal error: {error or "none"}.',
              'Small wave counts are descriptive and do not establish stable p95 latency under production load.']
    (args.out / 'summary.md').write_text('\n'.join(lines) + '\n')
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
