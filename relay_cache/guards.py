"""Frozen configuration, source/environment checks and exclusive GPU leases."""
import hashlib,json,os,platform,re,subprocess,tempfile
from pathlib import Path
from contextlib import contextmanager
from importlib import metadata
from .utils import sha256
SETTINGS=dict(method='relay_cache',stripe_width=8,history_budget=128,lengths=[256,512],
    block=32,threshold=.9,gamma=.8,track=4,mask=4,verify=False,seed=51713,precision='BF16',
    warmup='Every prompt has one excluded warm request immediately before timed replay',
    baseline_regenerated=False,algorithm_retuned=False)

def validate_settings(value):
    if value!=SETTINGS:raise ValueError("Frozen Relay settings changed; migration must not retune")

ROOT=Path(__file__).resolve().parent.parent

MANIFEST=ROOT/'dllm-eval/provenance/source_manifest.json'

PACKAGES=('torch','triton','transformers','huggingface-hub','numpy','sympy','math-verify','antlr4-python3-runtime','datasets','lm_eval')

def source_files(root=ROOT):
    return {str(p.relative_to(root)).replace('\\','/') for directory in ('relay_cache','dllm-eval/dllm_eval','third_party')
            for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json')}

def verify_sources(root=ROOT,manifest=None):
    declared=manifest or json.loads((root/'dllm-eval/provenance/source_manifest.json').read_text())
    if source_files(root)!=set(declared['files']):raise RuntimeError('Source file set changed')
    for name,digest in declared['files'].items():
        if sha256(root/name)!=digest:raise RuntimeError('Source hash mismatch: '+name)
    for name,digest in declared['artifacts'].items():
        if sha256(root/name)!=digest:raise RuntimeError('Frozen artifact mismatch: '+name)
    return declared

def environment():
    result=dict(python=platform.python_version(),platform=platform.platform(),packages={})
    for name in PACKAGES:
        try:result['packages'][name]=metadata.version(name)
        except metadata.PackageNotFoundError:result['packages'][name]=None
    return result

def verify_environment():
    expected=json.loads((ROOT/'dllm-eval/provenance/environment.json').read_text())
    if expected.get('status')!='captured':raise RuntimeError('Original evaluation environment has not been captured')
    actual=environment()
    for name,version in expected['packages'].items():
        if version is not None and actual['packages'].get(name)!=version:
            raise RuntimeError('Original environment version mismatch: '+name)
    if actual['python']!=expected['python']:raise RuntimeError('Original Python version mismatch')
    return actual

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_resume(saved,current):
    if saved!=current:raise RuntimeError('Run manifest changed; resume refused')

def inventory():
    command=['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']
    rows=[]
    for line in subprocess.check_output(command,text=True).splitlines():
        index,uuid,used,total,util=map(str.strip,line.split(','))
        rows.append(dict(index=int(index),uuid=uuid,used_mb=int(used),total_mb=int(total),utilization=int(util)))
    return rows

def select_idle(rows,requested,processes=()):
    matches=[r for r in rows if str(r['index'])==str(requested) or r['uuid']==requested]
    if len(matches)!=1:raise RuntimeError('GPU does not resolve uniquely')
    gpu=matches[0]
    if gpu['used_mb']>=128 or gpu['utilization']!=0 or gpu['total_mb']<24000:
        raise RuntimeError('Requested GPU is occupied or too small')
    if any(p['uuid']==gpu['uuid'] for p in processes):raise RuntimeError('GPU has a compute process')
    return gpu

def processes():
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return [dict(uuid=parts[0].strip(),pid=int(parts[1])) for line in lines.splitlines()
            if len(parts:=line.split(','))==2]

def check_binding():
    expected=os.environ.get('RELAY_GPU_UUID');visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not expected or not expected.startswith('GPU-') or visible!=expected:
        raise RuntimeError('Explicit Relay GPU UUID binding required')
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required per worker')
    props=torch.cuda.get_device_properties(0);actual=str(getattr(props,'uuid',expected))
    if actual.lower().removeprefix('gpu-')!=expected.lower().removeprefix('gpu-'):
        raise RuntimeError('Resolved GPU identity differs')
    return dict(visible_uuid=visible,actual_uuid=actual,device_name=props.name)

@contextmanager
def exclusive_lock(path):
    if os.name!='posix':raise RuntimeError('GPU execution requires Linux/POSIX flock')
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another job holds '+str(path)) from e
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

@contextmanager
def gpu_lease(requested):
    # Resolve first, then lock and recheck immediately before CUDA initialization.
    gpu=select_idle(inventory(),requested,processes())
    directory=Path(os.environ.get('RELAY_LOCK_DIR',str(Path(tempfile.gettempdir())/'relaycache-gpu-locks')))
    with exclusive_lock(directory/(gpu['uuid']+'.lock')):
        gpu=select_idle(inventory(),gpu['uuid'],processes())
        os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid'];os.environ['RELAY_GPU_UUID']=gpu['uuid']
        yield gpu
