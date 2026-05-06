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

DOCKER_IMAGE = 'pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel'
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


def rsync_to(state, local_path, remote_path):
    host, port = state['ssh_host'], state['ssh_port']
    subprocess.run([
        'rsync', '-avz', '--progress',
        '-e', f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}',
        str(local_path) + '/',
        f'root@{host}:{remote_path}/',
    ], check=True)


def rsync_from(state, remote_path, local_path):
    host, port = state['ssh_host'], state['ssh_port']
    os.makedirs(local_path, exist_ok=True)
    subprocess.run([
        'rsync', '-avz', '--update', '--progress',
        '-e', f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}',
        f'root@{host}:{remote_path}/',
        str(local_path) + '/',
    ], check=True)


def _require_instance(state):
    if not state.get('instance_id'):
        print("No instance. Run 'malign cloud launch' first.", file=sys.stderr)
        sys.exit(1)


# ── Commands ──────────────────────────────────────────────────────

def cmd_launch(args):
    state = load_state()
    if state.get('instance_id'):
        print(f"Instance already exists: {state['instance_id']}")
        print(f"SSH: ssh -p {state['ssh_port']} root@{state['ssh_host']}")
        print("Run 'malign cloud stop' first to destroy it.")
        return

    print("Searching for A100 80GB offers...", file=sys.stderr)
    raw = vastai(
        'search', 'offers',
        f'gpu_name=A100_SXM4 num_gpus=1 gpu_ram>={MIN_GPU_RAM} reliability>0.95 disk_space>={DISK_GB}',
        '-o', 'dph+',
        '--raw',
    )
    offers = json.loads(raw)
    if not offers:
        raw = vastai(
            'search', 'offers',
            f'gpu_name=A100 num_gpus=1 gpu_ram>={MIN_GPU_RAM} reliability>0.95 disk_space>={DISK_GB}',
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

    print(f"Best offer: #{offer_id} — {gpu} {ram}GB, ${price}/hr, {loc}")
    if not getattr(args, 'yes', False):
        confirm = input("Launch this instance? [y/N] ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return

    print("Creating instance...", file=sys.stderr)
    result = vastai(
        'create', 'instance', str(offer_id),
        '--image', DOCKER_IMAGE,
        '--disk', str(DISK_GB),
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
            parsed = eval(line)
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
        'gpu': f"{gpu} {ram}GB",
        'price_per_hour': float(price) if isinstance(price, (int, float)) else price,
        'launched_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
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

    print("Installing malign-logits...", file=sys.stderr)
    hf_login = f'huggingface-cli login --token {hf_token}' if hf_token else 'echo "No HF_TOKEN — skipping login"'
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
python -m spacy download en_core_web_sm
{hf_login}

python -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA {{torch.cuda.is_available()}}')"
python -c "from malign_logits import MODEL_FAMILIES; print(f'{{len(MODEL_FAMILIES)}} families registered')"

echo "SETUP COMPLETE"
"""
    ssh_run(state, setup_script)
    state['setup_done'] = True
    save_state(state)

    if LOCAL_STASH.exists():
        n_files = sum(1 for _ in LOCAL_STASH.rglob('*') if _.is_file())
        size_mb = sum(f.stat().st_size for f in LOCAL_STASH.rglob('*') if f.is_file()) / 1e6
        print(f"\nUploading local stash ({n_files} files, {size_mb:.0f} MB)...", file=sys.stderr)
        ssh_run(state, f'mkdir -p {REMOTE_STASH}')
        rsync_to(state, str(LOCAL_STASH), REMOTE_STASH)
        print(f"Stash uploaded.")

    print("\nSetup complete.")
    print("Next: malign cloud run")


def cmd_run(args):
    state = load_state()
    _require_instance(state)

    families_flag = f"--families {args.families}" if getattr(args, 'families', None) else ""
    skip_flag = f"--skip {args.skip}" if getattr(args, 'skip', '') else ""

    batch_cmd = (
        f'cd {REMOTE_REPO} && '
        f'HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null) '
        f'PYTHONUNBUFFERED=1 malign produce-all {families_flag} {skip_flag} '
        f'2>&1 | tee /workspace/produce-all.log'
    )

    session_name = 'produce_all'
    print(f"Starting produce-all in tmux session '{session_name}'...", file=sys.stderr)
    ssh_run(state, f"tmux kill-session -t {session_name} 2>/dev/null || true")
    ssh_run(state, f"tmux new-session -d -s {session_name} '{batch_cmd}'")

    state['running'] = session_name
    state['run_started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    save_state(state)

    print(f"\nBatch started.")
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

    if status == 'RUNNING':
        r = ssh_run(state, 'tail -10 /workspace/produce-all.log 2>/dev/null',
                    capture=True, check=False)
        if r.stdout.strip():
            print(f"\nLast log lines:")
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

    print("Downloading stash/ ...", file=sys.stderr)
    rsync_from(state, REMOTE_STASH, str(LOCAL_STASH))

    print(f"\nData and stash downloaded to {PROJECT_ROOT}")


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
    subprocess.run(['vastai', 'destroy', 'instance', instance_id],
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
    cmd = ssh_cmd(state) + [f'tail -{n} /workspace/produce-all.log 2>/dev/null || echo "No log found"']
    os.execvp(cmd[0], cmd)


def cmd_ssh(args):
    state = load_state()
    _require_instance(state)
    cmd = ssh_cmd(state)
    ssh_command = getattr(args, 'ssh_command', None)
    if ssh_command:
        cmd += [' '.join(ssh_command)]
    os.execvp(cmd[0], cmd)


def main(args):
    commands = {
        'launch': cmd_launch,
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
        print("Usage: malign cloud {launch|setup|run|status|download|stop|attach|log|ssh}")
        sys.exit(1)
    commands[cloud_cmd](args)
