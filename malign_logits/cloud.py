"""Vast.ai GPU instance manager for malign-logits data production.

Manages: launch → setup → run → status → download → stop.
State persisted in .vastai.json.

Prerequisites:
    pip install vastai
    vastai set api-key YOUR_KEY
    Upload SSH pubkey at https://cloud.vast.ai/manage-keys/
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / '.vastai.json'
REMOTE_WORK = '/workspace'
REMOTE_REPO = f'{REMOTE_WORK}/malign-logits'
REMOTE_DATA = f'{REMOTE_REPO}/data'
REMOTE_STASH = f'{REMOTE_REPO}/data/raw/stash'
LOCAL_STASH = PROJECT_ROOT / 'data' / 'raw' / 'stash'

DOCKER_IMAGE = 'vllm/vllm-openai:latest'
DISK_GB = 300
MIN_GPU_RAM = 79


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2) + '\n')


def vastai(*args, capture=True):
    cmd = ['vastai'] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"vastai error: {r.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        return r.stdout.strip()
    else:
        subprocess.run(cmd)


def ssh_cmd(state):
    return [
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        '-p', str(state['ssh_port']), f'root@{state["ssh_host"]}',
    ]


def ssh_run(state, command, check=True, capture=False):
    cmd = ssh_cmd(state) + [command]
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            print(f"SSH error: {r.stderr.strip()}", file=sys.stderr)
        return r
    else:
        r = subprocess.run(cmd)
        if check and r.returncode != 0:
            sys.exit(1)
        return r


def rsync_to(state, local_path, remote_path, exclude=None):
    host, port = state['ssh_host'], state['ssh_port']
    cmd = [
        'rsync', '-avz', '--progress',
        '-e', f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}',
    ]
    for pat in (exclude or []):
        cmd += ['--exclude', pat]
    cmd += [str(local_path) + '/', f'root@{host}:{remote_path}/']
    subprocess.run(cmd, check=True)


def rsync_from(state, remote_path, local_path, ignore_existing=False):
    host, port = state['ssh_host'], state['ssh_port']
    os.makedirs(local_path, exist_ok=True)
    cmd = [
        'rsync', '-avz', '--progress',
        '-e', f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}',
    ]
    if ignore_existing:
        cmd.append('--ignore-existing')
    cmd += [f'root@{host}:{remote_path}/', str(local_path) + '/']
    subprocess.run(cmd, check=True)


def _require_instance(state):
    if not state.get('instance_id'):
        print("No instance. Run 'malign cloud launch' first.", file=sys.stderr)
        sys.exit(1)


# ── Commands ──────────────────────────────────────────────────────

PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'data', 'cloud_profiles.json')


def load_profile(name):
    """A named machine shape + its package floors.

    Provisioning was retyped every time and the floors were REMEMBERED rather
    than declared -- which is how thirteen models in the July grid failed to
    load under torch<2.6 and read as a model problem instead of an environment
    one. A profile makes the floor an artifact.
    """
    if not os.path.exists(PROFILES_PATH):
        print(f"No profiles file at {PROFILES_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(PROFILES_PATH) as fh:
        profiles = json.load(fh)
    if name not in profiles or name.startswith('_'):
        avail = [k for k in profiles if not k.startswith('_')]
        print(f"Unknown profile {name!r}. Available: {', '.join(avail)}",
              file=sys.stderr)
        sys.exit(1)
    return profiles[name]


def cmd_profiles(args):
    with open(PROFILES_PATH) as fh:
        profiles = json.load(fh)
    for name, p in profiles.items():
        if name.startswith('_'):
            continue
        print(f"{name}")
        print(f"    {p.get('description','')}")
        print(f"    {p['num_gpus']}x {p['gpu_name']} >={p['min_gpu_ram']}GB, "
              f"{p['disk_gb']}GB disk, {p['image']}")
        if p.get('pins'):
            print(f"    pins: {', '.join(p['pins'])}")
        for pin, why in (p.get('pin_reasons') or {}).items():
            print(f"        {pin}: {why}")


def cmd_launch(args):
    state = load_state()
    if state.get('instance_id'):
        print(f"Instance already exists: {state['instance_id']}")
        print(f"SSH: ssh -p {state['ssh_port']} root@{state['ssh_host']}")
        print("Run 'malign cloud stop' first to destroy it.")
        return

    prof = load_profile(getattr(args, 'profile', None) or 'default')
    num_gpus = getattr(args, 'num_gpus', None) or prof['num_gpus']
    disk_gb = getattr(args, 'disk', None) or prof['disk_gb']
    gpu_name = prof['gpu_name']
    min_ram = prof['min_gpu_ram']
    image = prof['image']
    rel = prof.get('min_reliability', 0.95)
    cuda = prof.get('cuda_max_good', 12.4)
    #: COMPUTE PARALLELISES ACROSS GPUs; THE NETWORK DOES NOT. A roster is
    #: ~1.4 TB of weights that must cross one link no matter how many cards sit
    #: behind it, so a cheap box with a slow NIC is not cheap. See
    #: data/cloud_profiles.json _bandwidth_note for the measured case.
    min_down = prof.get('min_inet_down_mbps', 0)
    print(f"Profile {getattr(args, 'profile', None) or 'default'}: "
          f"{num_gpus}x {gpu_name} >={min_ram}GB, {disk_gb}GB, {image}, "
          f"link >={min_down} Mbps", file=sys.stderr)
    if prof.get('pins'):
        print(f"  package floors: {', '.join(prof['pins'])}", file=sys.stderr)

    print(f"Searching for {num_gpus}× A100 80GB offers ({disk_gb} GB disk)...", file=sys.stderr)
    raw = vastai(
        'search', 'offers',
        f'gpu_name={gpu_name} num_gpus={num_gpus} gpu_ram>={min_ram} reliability>{rel} disk_space>={disk_gb} cuda_max_good>={cuda} inet_down>={min_down}',
        '-o', 'dph+',
        '--raw',
    )
    offers = json.loads(raw)
    if not offers:
        raw = vastai(
            'search', 'offers',
            f'gpu_name=A100 num_gpus={num_gpus} gpu_ram>={min_ram} reliability>{rel} disk_space>={disk_gb} cuda_max_good>={cuda} inet_down>={min_down}',
            '-o', 'dph+',
            '--raw',
        )
        offers = json.loads(raw)

    if not offers:
        print("No suitable offers found.", file=sys.stderr)
        sys.exit(1)

    offer = offers[0]
    offer_id = offer['id']
    price = offer.get('dph_total', offer.get('dph', '?'))
    gpu = offer.get('gpu_name', '?')
    ram = offer.get('gpu_ram', '?')
    loc = offer.get('geolocation', '?')

    n_gpu = offer.get('num_gpus', num_gpus)
    print(f"Best offer: #{offer_id} — {n_gpu}× {gpu} {ram}GB, ${price}/hr, {loc}")
    if not getattr(args, 'yes', False):
        confirm = input("Launch this instance? [y/N] ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return

    print("Creating instance...", file=sys.stderr)
    result = vastai(
        'create', 'instance', str(offer_id),
        '--image', image,
        '--disk', str(disk_gb),
        '--ssh',
        '--direct',
    )
    print(result)

    import re
    instance_id = None
    for line in result.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and parsed.get('new_contract'):
                instance_id = str(parsed['new_contract'])
                break
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            import ast
            parsed = ast.literal_eval(line)  # vast.ai emits Python dict-repr (single quotes)
            if isinstance(parsed, dict) and parsed.get('new_contract'):
                instance_id = str(parsed['new_contract'])
                break
        except Exception:
            pass
    if not instance_id:
        match = re.search(r"'new_contract':\s*(\d+)", result)
        if match:
            instance_id = match.group(1)
    if not instance_id:
        for word in result.split():
            if word.isdigit() and len(word) >= 6:
                instance_id = word
                break
    if not instance_id:
        print("Could not parse instance ID.", file=sys.stderr)
        sys.exit(1)

    print(f"Instance {instance_id} created. Waiting for SSH...", file=sys.stderr)

    ssh_host, ssh_port = None, None
    status = ''  # bind before use: the else branch below references it even on
                 # the first poll, when no instance may match instance_id yet
    for attempt in range(60):
        raw = vastai('show', 'instances', '--raw')
        instances = json.loads(raw)
        for inst in instances:
            if str(inst.get('id')) == instance_id:
                status = inst.get('actual_status', inst.get('status', ''))
                ssh_host = inst.get('ssh_host')
                ssh_port = inst.get('ssh_port')
                if status == 'running' and ssh_host and ssh_port:
                    break
        else:
            if attempt % 6 == 0:
                print(f"  Waiting... ({attempt * 5}s, status={status})", file=sys.stderr)
            time.sleep(5)
            continue
        break
    else:
        print("Timed out waiting for instance.", file=sys.stderr)
        state['instance_id'] = instance_id
        save_state(state)
        sys.exit(1)

    state = {
        'instance_id': instance_id,
        'offer_id': str(offer_id),
        'ssh_host': ssh_host,
        'ssh_port': int(ssh_port),
        'num_gpus': num_gpus,
        'gpu': f"{n_gpu}× {gpu} {ram}GB",
        'price_per_hour': float(price) if isinstance(price, (int, float)) else price,
        'launched_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        # THE PROFILE TRAVELS WITH THE INSTANCE. cmd_setup applies its package
        # floors; a profile that only shaped the SEARCH would leave the floors
        # as prose, which is the state that cost the July grid 13 models.
        'profile': getattr(args, 'profile', None) or 'default',
        'pins': prof.get('pins', []),
        'image': image,
        #: the environment record must be able to say what the link WAS: a cost
        #: estimate that does not name its bandwidth is not reproducible.
        'inet_down_mbps': offer.get('inet_down'),
        'min_inet_down_mbps': min_down,
    }
    save_state(state)

    print(f"\nInstance {instance_id} running!")
    print(f"SSH: ssh -p {ssh_port} root@{ssh_host}")
    print(f"Cost: ${price}/hr")
    print(f"\nNext: malign cloud setup")


def cmd_setup(args):
    state = load_state()
    _require_instance(state)

    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        print("HF_TOKEN found locally — will configure on remote.", file=sys.stderr)
    else:
        print("Warning: HF_TOKEN not set. Gated models (Llama, etc.) will fail.", file=sys.stderr)

    pins = state.get('pins') or load_profile(state.get('profile', 'default')).get('pins', [])
    if pins:
        print(f"Profile {state.get('profile','default')} pins: {', '.join(pins)}",
              file=sys.stderr)
    # A FLOOR IS NOT AN UPGRADE, AND `pip install -U` CANNOT TELL THE DIFFERENCE.
    #
    # ON INSTANCE 46670110, 2026-08-03. MEASURED: the vllm image ships a
    # MATCHED CUDA SET and `vllm` declares it EXACTLY -- `torch==2.11.0`,
    # `torchvision==0.26.0`, `torchaudio==2.11.0`; the torchvision dist-info
    # carries the image's build date. After `malign cloud setup` the box held
    # a generic PyPI **torch 2.13.0 with no +cu130 build** beside
    # torchvision 0.26.0+cu130. That desynced pair is measured; the failure
    # surfaced four layers away as
    #
    #     RuntimeError: operator torchvision::nms does not exist
    #
    # while importing FalconH1 -- which reads as "this model is unsupported"
    # and is really "this environment was broken by its own provisioning". The
    # repair was `pip install torch==2.11.0+cu130` from the cu130 index: BACK
    # DOWN to what the image shipped, which satisfied the floor all along.
    #
    # **WHICH COMMAND DID THE UPGRADING IS INFERRED, NOT MEASURED, AND THE BOX
    # CAN NO LONGER SETTLE IT** -- the repair changed the state before the
    # question was asked. Setup runs two things that mention torch: this pin,
    # and `pip install -e .` against `requirements.txt` (`torch>=2.6.0`). The
    # second carries no `-U`, so at 2.11.0 it is a no-op; **this line held the
    # only `-U` in the path.** That is a strong inference and it is not a
    # measurement, so it is written as one.
    #
    # **THE PIN EXISTS TO RAISE torch OFF 2.5.1, NOT TO CHASE LATEST.** So test
    # first and install only when the floor is genuinely unmet. An unmet floor
    # still installs exactly as before; a met one becomes a no-op, which is
    # what "floor" meant.
    # THE FIX IS DROPPING `-U`. Plain `pip install 'torch>=2.6'` reports
    # "Requirement already satisfied" and does nothing when the floor holds,
    # and installs when it does not -- which is what a floor means. `-U` was
    # doing the one thing a floor must never do.
    pin_cmd = ("pip install " + " ".join(f"'{p}'" for p in pins)) if pins \
        else 'echo "no package floors declared for this profile"'

    print("Installing malign-logits...", file=sys.stderr)
    hf_login = f'(hf auth login --token {hf_token} 2>/dev/null || huggingface-cli login --token {hf_token} 2>/dev/null || echo "HF login failed — gated models may not work")' if hf_token else 'echo "No HF_TOKEN — skipping login"'
    setup_script = f"""
set -ex
which python || ln -sf $(which python3) /usr/local/bin/python

if [ ! -d {REMOTE_REPO} ]; then
    git clone https://github.com/quadrismegistus/malign-logits.git {REMOTE_REPO}
else
    cd {REMOTE_REPO} && git pull
fi

cd {REMOTE_REPO}
pip install -e .

# PACKAGE FLOORS FROM THE PROFILE, applied AFTER `pip install -e .` so a
# transitive downgrade cannot undo them. transformers refuses .bin checkpoints
# under torch<2.6 and the failure reads as a broken model, not a broken box.
{pin_cmd}
python - <<'PYCHECK'
import torch, transformers
print(f"ENVIRONMENT RECORD  torch {{torch.__version__}}  "
      f"transformers {{transformers.__version__}}  cuda {{torch.version.cuda}}")
maj, mino = (torch.__version__.split('.') + ['0'])[:2]
if (int(maj), int(''.join(c for c in mino if c.isdigit()) or 0)) < (2, 6):
    raise SystemExit("REFUSING: torch %s is below the 2.6 floor; .bin "
                     "checkpoints will fail to load." % torch.__version__)
PYCHECK

python -m spacy download en_core_web_sm
{hf_login}

python -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA {{torch.cuda.is_available()}}')"
python -c "from malign_logits import MODEL_FAMILIES; print(f'{{len(MODEL_FAMILIES)}} families registered')"

echo "SETUP COMPLETE"
"""
    ssh_run(state, setup_script)
    state['setup_done'] = True
    save_state(state)

    local_data = PROJECT_ROOT / 'data'
    if local_data.exists():
        exclude = ['stash_gen_metrics', 'stash', 'stash_gen_battery', 'stash_self_surprisal']
        size_mb = sum(f.stat().st_size for f in local_data.rglob('*')
                      if f.is_file() and not any(x in str(f) for x in exclude)) / 1e6
        print(f"\nUploading data/ ({size_mb:.0f} MB, excluding old stashes)...",
              file=sys.stderr)
        rsync_to(state, str(local_data), REMOTE_DATA, exclude=exclude)
        print(f"Data uploaded.")

    print("\nSetup complete.")
    print("Next: malign cloud run")


def cmd_run(args):
    state = load_state()
    _require_instance(state)

    custom_cmd = getattr(args, 'command', None)
    if custom_cmd:
        # Arbitrary command mode
        user_cmd = ' '.join(custom_cmd)
        session_name = user_cmd.split()[0].split('/')[-1].replace('.py', '')[:20]
        log_file = f'/workspace/{session_name}.log'
        batch_cmd = (
            f'cd {REMOTE_REPO} && git pull && '
            f'HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null) '
            f'PYTHONUNBUFFERED=1 {user_cmd} '
            f'2>&1 | tee {log_file}'
        )
    else:
        # Default: produce-all
        families_flag = f"--families {args.families}" if getattr(args, 'families', None) else ""
        skip_flag = f"--skip {args.skip}" if getattr(args, 'skip', '') else ""
        session_name = 'produce_all'
        log_file = '/workspace/produce-all.log'
        batch_cmd = (
            f'cd {REMOTE_REPO} && git pull && '
            f'HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null) '
            f'PYTHONUNBUFFERED=1 malign produce-all {families_flag} {skip_flag} '
            f'2>&1 | tee {log_file}'
        )

    print(f"Starting '{session_name}' in tmux...", file=sys.stderr)
    print(f"  Command: {batch_cmd[:120]}...", file=sys.stderr)
    ssh_run(state, f"tmux kill-session -t {session_name} 2>/dev/null || true")
    # Write command to a script to avoid shell escaping issues in tmux
    ssh_run(state, f"cat > /workspace/run_{session_name}.sh << 'RUNEOF'\n{batch_cmd}\nRUNEOF")
    ssh_run(state, f"chmod +x /workspace/run_{session_name}.sh")
    ssh_run(state, f"tmux new-session -d -s {session_name} 'bash /workspace/run_{session_name}.sh'")

    state['running'] = session_name
    state['log_file'] = log_file
    state['run_started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    save_state(state)

    print(f"\nStarted.")
    print(f"Monitor: malign cloud status")
    print(f"Attach:  malign cloud attach")


def cmd_status(args):
    state = load_state()
    _require_instance(state)

    print(f"Instance: {state['instance_id']}")
    print(f"GPU: {state.get('gpu', '?')}")
    print(f"SSH: ssh -p {state['ssh_port']} root@{state['ssh_host']}")

    price = state.get('price_per_hour', 0)
    if price and state.get('launched_at'):
        from datetime import datetime
        launched = datetime.fromisoformat(state['launched_at'])
        hours = (datetime.now() - launched).total_seconds() / 3600
        print(f"Running: {hours:.1f}h, est. cost: ${hours * price:.2f}")

    session_name = state.get('running', 'produce_all')
    r = ssh_run(state, f"tmux has-session -t {session_name} 2>/dev/null && echo RUNNING || echo STOPPED",
                capture=True, check=False)
    status = r.stdout.strip()
    print(f"\nBatch: {status}")

    log_file = state.get('log_file', '/workspace/produce-all.log')
    if status == 'RUNNING':
        r = ssh_run(state, f'tail -10 {log_file} 2>/dev/null',
                    capture=True, check=False)
        if r.stdout.strip():
            print(f"\nLast log lines ({log_file}):")
            for line in r.stdout.strip().split('\n'):
                print(f"  {line}")

    r = ssh_run(state, f'ls {REMOTE_DATA}/*.csv 2>/dev/null | wc -l',
                capture=True, check=False)
    n_csv = r.stdout.strip() if r.returncode == 0 else '0'
    print(f"\nCSV files produced: {n_csv}")


def cmd_download(args):
    state = load_state()
    _require_instance(state)

    print("Downloading data/ ...", file=sys.stderr)
    rsync_from(state, REMOTE_DATA, str(PROJECT_ROOT / 'data'))

    print("Downloading figures/ ...", file=sys.stderr)
    rsync_from(state, f'{REMOTE_REPO}/figures', str(PROJECT_ROOT / 'figures'))

    print(f"\nData + figures downloaded to {PROJECT_ROOT}")


def cmd_stop(args):
    state = load_state()
    _require_instance(state)

    instance_id = state['instance_id']
    price = state.get('price_per_hour', 0)
    if price and state.get('launched_at'):
        from datetime import datetime
        launched = datetime.fromisoformat(state['launched_at'])
        hours = (datetime.now() - launched).total_seconds() / 3600
        print(f"Instance {instance_id} running {hours:.1f}h, est. cost: ${hours * price:.2f}")

    if not getattr(args, 'yes', False):
        confirm = input("Destroy this instance? (data will be lost) [y/N] ").strip().lower()
        if confirm != 'y':
            print("Aborted. Run 'malign cloud download' first.")
            return

    print(f"Destroying instance {instance_id}...", file=sys.stderr)
    #: str(): the state file stores instance_id as an INT (json round-trips
    #: the number vastai returns), and subprocess rejects a non-str argv
    #: entry with `TypeError: expected str, bytes or os.PathLike object,
    #: not int`. Every OTHER call site formats it into a string first, so
    #: this was the one path that never ran -- and it is the teardown
    #: path, which is the one that costs money when it fails.
    subprocess.run(['vastai', 'destroy', 'instance', str(instance_id)],
                   input='y\n', text=True, capture_output=True)
    STATE_FILE.unlink(missing_ok=True)
    print("Instance destroyed. All billing stopped.")


def cmd_attach(args):
    state = load_state()
    _require_instance(state)
    cmd = ssh_cmd(state) + ['-t', 'tmux attach -t produce_all 2>/dev/null || tmux attach']
    os.execvp(cmd[0], cmd)


def cmd_log(args):
    state = load_state()
    _require_instance(state)
    n = getattr(args, 'lines', 30) or 30
    log_file = state.get('log_file', '/workspace/produce-all.log')
    cmd = ssh_cmd(state) + [f'tail -{n} {log_file} 2>/dev/null || echo "No log found"']
    os.execvp(cmd[0], cmd)


def cmd_ssh(args):
    state = load_state()
    _require_instance(state)
    cmd = ssh_cmd(state)
    ssh_command = getattr(args, 'ssh_command', None)
    if ssh_command:
        cmd += [' '.join(ssh_command)]
    os.execvp(cmd[0], cmd)


def cmd_coverage(args):
    """PLANNED vs DELIVERED for a run directory, at BOTH units.

    This exists because `data/grid_run_manifest.json` records `cells: 100837`
    and that is the PLAN. 91,421 rows were on disk. The two numbers differ by
    a Falcon-shaped hole, and for weeks the roster figure was the one quoted.

    Reports at the MODEL unit and the LINEAGE unit, because they differ and
    the second is the one paired analysis uses: the July run lost 9.7% of
    models and 12.9% of lineages, since the missing models clustered into
    whole lineages rather than spreading across them.
    """
    import glob as _glob
    from collections import Counter

    run_dir = args.run_dir
    files = sorted(_glob.glob(os.path.join(run_dir, "*.jsonl")))
    if not files:
        print(f"No .jsonl shards in {run_dir}", file=sys.stderr)
        sys.exit(1)

    spec_path = args.spec
    if not os.path.isabs(spec_path):
        spec_path = os.path.join(os.path.dirname(PROFILES_PATH), '..', spec_path)
        spec_path = os.path.normpath(spec_path)
    roster, expect = [], args.expect
    if os.path.exists(spec_path):
        raw = json.load(open(spec_path))
        entries = raw["spec"] if isinstance(raw, dict) else raw
        roster = [e["model"] for e in entries]
        if expect is None and entries:
            expect = max(len(e["prompts"]) for e in entries)
    if expect is None:
        print("Cannot infer rows-per-model; pass --expect", file=sys.stderr)
        sys.exit(1)

    counts = {}
    for f in files:
        model = os.path.basename(f)[:-6].replace("__", "/")
        with open(f, errors="ignore") as fh:
            counts[model] = sum(1 for _ in fh)

    complete = {m for m, n in counts.items() if n >= expect}
    partial = {m: n for m, n in counts.items() if 0 < n < expect}
    absent = [m for m in roster if m not in counts] if roster else []
    rows = sum(counts.values())
    planned = len(roster) * expect if roster else None

    out = []
    def P(line=""):
        out.append(line)
        print(line)

    P(f"COVERAGE — {run_dir}")
    P(f"    spec        {spec_path}")
    P(f"    expect      {expect} rows per complete model")
    P()
    if roster:
        P(f"    ROSTER                {len(roster)} models x {expect} = "
          f"{planned:,} PLANNED")
    P(f"    COMPLETE              {len(complete)}")
    P(f"    PARTIAL               {len(partial)}")
    P(f"    ABSENT ENTIRELY       {len(absent)}")
    P(f"    ROWS ON DISK          {rows:,}" +
      (f"   = {100.0 * rows / planned:.1f}% of planned" if planned else ""))
    if planned and rows != planned:
        P()
        P("    The roster figure is the PLAN, not the achievement. Anyone "
          "quoting")
        P(f"    {planned:,} as a delivered count is quoting the roster.")

    if partial:
        P()
        P("    PARTIALS, named:")
        for m, n in sorted(partial.items(), key=lambda kv: kv[1]):
            P(f"        {m:<52} {n:>6} / {expect}")
    if absent:
        P()
        P("    ABSENT, named:")
        for m in absent:
            P(f"        {m}")

    # ── the LINEAGE unit ────────────────────────────────────────────────
    lm_path = os.path.join(os.path.dirname(PROFILES_PATH),
                           "lineage_map_models.json")
    if os.path.exists(lm_path) and roster:
        lm = json.load(open(lm_path)).get("model_to_lineage", {})
        unmapped = [m for m in roster if m not in lm]
        lin_all = {lm[m] for m in roster if m in lm}
        lin_ok = {lm[m] for m in complete if m in lm}
        lost = sorted(lin_all - lin_ok)
        P()
        P(f"    UNIT = model      roster {len(roster)}  complete "
          f"{len(complete)}  partial {len(partial)}  absent {len(absent)}")
        P(f"    UNIT = lineage    roster {len(lin_all)}  complete "
          f"{len(lin_ok)}  lost {len(lost)}")
        if unmapped:
            P(f"    {len(unmapped)} roster model(s) have NO lineage entry and "
              f"are excluded from the lineage counts:")
            for m in unmapped[:10]:
                P(f"        {m}")
        if lost:
            P("    lineages LOST ENTIRELY (no complete member):")
            for L in lost:
                P(f"        {L}")
        if roster and lin_all:
            ml = 100.0 * (len(roster) - len(complete)) / len(roster)
            ll = 100.0 * len(lost) / len(lin_all)
            P(f"    model-level loss {ml:.1f}%   lineage-level loss {ll:.1f}%"
              + ("   <- WORSE at the unit paired analysis uses"
                 if ll > ml else ""))

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"\nwrote {args.out}")


def main(args):
    commands = {
        'launch': cmd_launch,
        'profiles': cmd_profiles,
        'coverage': cmd_coverage,
        'setup': cmd_setup,
        'run': cmd_run,
        'status': cmd_status,
        'download': cmd_download,
        'stop': cmd_stop,
        'attach': cmd_attach,
        'log': cmd_log,
        'ssh': cmd_ssh,
    }
    cloud_cmd = getattr(args, 'cloud_command', None)
    if not cloud_cmd:
        print("Usage: malign cloud {launch|profiles|setup|run|status|coverage|download|stop|attach|log|ssh}")
        sys.exit(1)
    commands[cloud_cmd](args)
