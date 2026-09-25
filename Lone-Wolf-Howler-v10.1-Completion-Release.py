#!/usr/bin/env python3
"""Lone-Wolf-Howler v10.1 — recovery/completion GGUF release pipeline.

Starts from the already validated F16 and Q4_K_M GGUF artifacts; does not retrain.
Each remaining quantization is generated from F16, structurally checked, CUDA-smoke-tested,
published to Hugging Face, recorded in resumable state, then deleted locally.
"""
from __future__ import annotations
import os, json, time, hashlib, shutil, subprocess, platform
from pathlib import Path

HF_USER=os.environ.get("HF_USER","aryansexter")
HF_REPO=os.environ.get("HF_REPO","Lone_Wolf_Howler_Rore")
HF_REPO_ID=f"{HF_USER}/{HF_REPO}"
RUNTIME_REPO=os.environ.get("RUNTIME_HF_REPO","Lone_Wolf_Howler_Rore-Runtime")
RUNTIME_REPO_ID=f"{HF_USER}/{RUNTIME_REPO}"
SCRATCH="/kaggle/tmp/lone_wolf_howler_scratch"
GGUF_DIR=f"{SCRATCH}/gguf_quantized_models"
LLAMA=f"{SCRATCH}/llama.cpp/build_cuda/bin/llama-cli"
QUANT=f"{SCRATCH}/llama.cpp/build_cuda/bin/llama-quantize"
F16=f"{GGUF_DIR}/lone-wolf-howler-f16.gguf"
Q4=f"{GGUF_DIR}/lone-wolf-howler-q4_k_m.gguf"
STATE_DIR="/kaggle/working/lone_wolf_howler_state"
STATE=f"{STATE_DIR}/gguf_stage_state.json"
MANIFEST=f"{STATE_DIR}/PUBLICATION_MANIFEST.json"
QUANTS=["Q8_0","Q6_K","Q5_K_M","Q5_K_S","Q4_K_M","Q4_0","Q3_K_M","Q2_K","IQ4_XS"]
REMAINING=[q for q in QUANTS if q!="Q4_K_M"]
ENV=os.environ.copy(); ENV["CUDA_VISIBLE_DEVICES"]="0"; ENV["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; ENV.pop("GGML_CUDA_DEVICES",None)
SMOKE_TIMEOUT=int(os.environ.get("GGUF_SMOKE_TIMEOUT","300"))
FORCE=os.environ.get("FORCE_REQUANTIZE","0")=="1"

def log(x=""): print(x,flush=True)
def banner(x): log("\n"+"="*78+"\n"+x+"\n"+"="*78)
def free_gb():
    u=shutil.disk_usage("/kaggle"); return u.free/1024**3
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def state():
    os.makedirs(STATE_DIR,exist_ok=True)
    try:
        with open(STATE,"r",encoding="utf-8") as f: s=json.load(f)
        s.setdefault("version",2); s.setdefault("stages",{}); return s
    except Exception: return {"version":2,"stages":{}}
def save(s):
    t=STATE+".tmp"
    with open(t,"w",encoding="utf-8") as f: json.dump(s,f,indent=2)
    os.replace(t,STATE)
def mark(q,status,path=None,error=None,**extra):
    s=state(); item={"status":status,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S")}
    if path:
        item["artifact"]=os.path.abspath(path)
        if os.path.isfile(path): item.update(size_bytes=os.path.getsize(path),sha256=sha256(path))
    if error: item["error"]=str(error)
    item.update(extra); s["stages"][q]=item; save(s); log(f"[STATE] {q} -> {status}")
def token():
    t=os.environ.get("HF_TOKEN","").strip()
    if not t:
        try:
            from kaggle_secrets import UserSecretsClient
            t=UserSecretsClient().get_secret("HF_TOKEN").strip()
        except Exception: pass
    if not t: raise RuntimeError("HF_TOKEN is missing. Add it as a Kaggle Secret or environment variable.")
    return t
def api():
    from huggingface_hub import HfApi
    return HfApi(token=token())
def remote_files(a):
    i=a.repo_info(repo_id=HF_REPO_ID,repo_type="model",files_metadata=True)
    return {x.rfilename:x for x in (getattr(i,"siblings",[]) or []) if getattr(x,"rfilename",None)}
def upload(a,local,remote):
    last=None
    for n in range(1,4):
        try:
            log(f"[HF] upload {n}/3: {remote}")
            a.upload_file(path_or_fileobj=local,path_in_repo=remote,repo_id=HF_REPO_ID,repo_type="model")
            return
        except Exception as e:
            last=e; log(f"[HF] warning: {e}"); time.sleep(5*n)
    raise RuntimeError(f"HF upload failed: {remote}") from last
def verify_remote(a,remote,local=None):
    f=remote_files(a)
    if remote not in f: raise RuntimeError(f"Remote file missing: {remote}")
    rs=getattr(f[remote],"size",None)
    if local and rs is not None and int(rs)!=os.path.getsize(local): raise RuntimeError(f"Remote size mismatch: {remote}")
def structural(p):
    if not os.path.isfile(p) or os.path.getsize(p)<1024*1024: raise RuntimeError(f"Invalid/missing GGUF: {p}")
    with open(p,"rb") as f: magic=f.read(4)
    if magic!=b"GGUF": raise RuntimeError(f"Bad GGUF magic: {p}")
    log(f"[GGUF] PASS {os.path.basename(p)} ({os.path.getsize(p)/1024**2:.1f} MiB)")
def smoke(p,label="RELEASE_OK"):
    banner(f"CUDA SMOKE: {os.path.basename(p)}")
    cmd=[LLAMA,"-m",p,"--device","CUDA0","-ngl","99","-c","128","-b","16","-t","2","-n","16","-p",f"Reply with exactly: {label}","--no-warmup","-st"]
    r=subprocess.run(cmd,env=ENV,text=True,capture_output=True,timeout=SMOKE_TIMEOUT,check=False)
    out=(r.stdout or "")+"\n"+(r.stderr or "")
    log(out[-6000:])
    low=out.lower()
    if r.returncode!=0: raise RuntimeError(f"Smoke failed: {r.returncode}")
    for bad in ("failed to initialize cuda","no cuda-capable device is detected","no usable gpu found","gpu-layers option will be ignored"):
        if bad in low: raise RuntimeError(f"CUDA failure reported: {bad}")
    if not any(x in low for x in ("cuda","ggml_cuda","offload","gpu")): raise RuntimeError("No CUDA/GPU evidence in smoke output")
    log("[SMOKE] PASS")
def ensure_f16(a):
    if os.path.isfile(F16): structural(F16); return F16
    banner("RECOVER F16 FROM HF")
    if free_gb()<9: raise RuntimeError(f"Only {free_gb():.2f} GB free; need >=9 GB to recover F16")
    from huggingface_hub import hf_hub_download
    files=remote_files(a); remote=next((x for x in ("gguf/lone-wolf-howler-f16.gguf","fp16_model/gguf_output/lone-wolf-howler-f16.gguf") if x in files),None)
    if not remote: raise RuntimeError("F16 not local and not found on HF")
    p=hf_hub_download(repo_id=HF_REPO_ID,filename=remote,repo_type="model",local_dir=GGUF_DIR,local_dir_use_symlinks=False)
    os.makedirs(GGUF_DIR,exist_ok=True)
    if p!=F16: shutil.copy2(p,F16)
    structural(F16); return F16
def recover(a):
    banner("RECOVER HF PUBLICATION STATE")
    files=remote_files(a); s=state(); count=0
    paths={"F16":["gguf/lone-wolf-howler-f16.gguf","fp16_model/gguf_output/lone-wolf-howler-f16.gguf"]}
    for q in QUANTS: paths[q]=[f"gguf/lone-wolf-howler-{q.lower()}.gguf"]
    for q,cands in paths.items():
        if s["stages"].get(q,{}).get("status")=="PUBLISHED": continue
        found=next((p for p in cands if p in files),None)
        if found:
            s["stages"][q]={"status":"PUBLISHED","remote_path":found,"recovered_from_hf":True,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S")}; count+=1; log(f"[RECOVERY] {q}: {found}")
    save(s); log(f"[RECOVERY] {count} stage(s) recovered")
def publish_f16(a,p):
    upload(a,p,"gguf/lone-wolf-howler-f16.gguf"); verify_remote(a,"gguf/lone-wolf-howler-f16.gguf",p)
    upload(a,p,"fp16_model/gguf_output/lone-wolf-howler-f16.gguf"); verify_remote(a,"fp16_model/gguf_output/lone-wolf-howler-f16.gguf",p)
    mark("F16","PUBLISHED",p)
def publish_quant(a,q,p):
    remote=f"gguf/lone-wolf-howler-{q.lower()}.gguf"; upload(a,p,remote); verify_remote(a,remote,p); mark(q,"PUBLISHED",p,remote_path=remote)
def quantize(f16,q,out):
    if os.path.isfile(out) and not FORCE: structural(out); return
    if os.path.isfile(out): os.remove(out)
    mark(q,"QUANTIZING",out)
    subprocess.run([QUANT,f16,out,q],check=True)
    structural(out)
def manifest(a):
    s=state(); files=remote_files(a); payload={"model":"Lone-Wolf-Howler","hf_repo":HF_REPO_ID,"generated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"cuda":{"visible_devices":"0","device_order":"PCI_BUS_ID","device":"CUDA0","gpu_layers":99,"single_turn":True},"quantizations":{},"stages":s["stages"]}
    for q in ["F16"]+QUANTS:
        r="gguf/lone-wolf-howler-f16.gguf" if q=="F16" else f"gguf/lone-wolf-howler-{q.lower()}.gguf"
        payload["quantizations"][q]={"remote_path":r,"present":r in files,"size_bytes":getattr(files[r],"size",None) if r in files else None}
    Path(MANIFEST).write_text(json.dumps(payload,indent=2),encoding="utf-8"); upload(a,MANIFEST,"PUBLICATION_MANIFEST.json")
def readme(a):
    s=state(); rows=[]
    for q in ["F16"]+QUANTS: rows.append(f"| `{q}` | {s['stages'].get(q,{}).get('status','PENDING')} |")
    text="""# Lone-Wolf-Howler\n\nValidated GGUF release produced by the recovery/completion pipeline.\n\n## Quantizations\n\n| Quantization | Status |\n|---|---|\n"""+"\n".join(rows)+"""\n\n## CUDA validation\n\n- CUDA_VISIBLE_DEVICES=0\n- CUDA_DEVICE_ORDER=PCI_BUS_ID\n- llama.cpp device: CUDA0\n- GPU layers: 99\n- single-turn mode: -st\n\nF16 and Q4_K_M were validated before continuation. Remaining quantizations are smoke-tested before publication.\n\nThe pipeline records state after each stage and can be safely rerun after interruption.\n"""
    p=f"{STATE_DIR}/README.md"; Path(p).write_text(text,encoding="utf-8"); upload(a,p,"README.md")
def runtime_bundle(a):
    if os.environ.get("PUBLISH_RUNTIME_BUNDLE","1")!="1": return
    root=Path(STATE_DIR)/"runtime_bundle"; env=root/"environment"; env.mkdir(parents=True,exist_ok=True); (root/"llama.cpp").mkdir(exist_ok=True)
    (env/"python_version.txt").write_text(__import__("sys").version,encoding="utf-8")
    r=subprocess.run([__import__("sys").executable,"-m","pip","freeze"],capture_output=True,text=True,check=False); (env/"pip_freeze.txt").write_text(r.stdout,encoding="utf-8")
    r=subprocess.run(["nvidia-smi"],capture_output=True,text=True,check=False); (env/"nvidia_smi.txt").write_text((r.stdout or "")+(r.stderr or ""),encoding="utf-8")
    (root/"runtime_manifest.json").write_text(json.dumps({"model_repository":HF_REPO_ID,"runtime_repository":RUNTIME_HF_REPO_ID,"llama_cli":LLAMA,"llama_quantize":QUANT,"credentials_saved":False,"generated_at":time.strftime("%Y-%m-%dT%H:%M:%S")},indent=2),encoding="utf-8")
    for p in (LLAMA,QUANT):
        if os.path.isfile(p): shutil.copy2(p,root/"llama.cpp"/os.path.basename(p))
    from huggingface_hub import HfApi
    rt=HfApi(token=token()); rt.create_repo(repo_id=RUNTIME_HF_REPO_ID,repo_type="model",exist_ok=True)
    for base,_,names in os.walk(root):
        for n in names:
            local=os.path.join(base,n); remote=os.path.relpath(local,root).replace(os.sep,"/")
            try: rt.upload_file(path_or_fileobj=local,path_in_repo=remote,repo_id=RUNTIME_HF_REPO_ID,repo_type="model")
            except Exception as e: log(f"[runtime] warning {remote}: {e}")
def final_verify(a):
    files=remote_files(a); required=["README.md","PUBLICATION_MANIFEST.json","gguf/lone-wolf-howler-f16.gguf","fp16_model/gguf_output/lone-wolf-howler-f16.gguf"]+[f"gguf/lone-wolf-howler-{q.lower()}.gguf" for q in QUANTS]
    missing=[p for p in required if p not in files]
    if missing: raise RuntimeError("Missing remote artifacts:\n"+"\n".join(missing))
    s=state(); missing_state=[q for q in ["F16"]+QUANTS if s["stages"].get(q,{}).get("status")!="PUBLISHED"]
    if missing_state: raise RuntimeError("State incomplete: "+", ".join(missing_state))
    for p in required: log(f"[VERIFY] {p}: {getattr(files[p],'size',None)} bytes")
def main():
    banner("LONE-WOLF-HOWLER COMPLETION / RECOVERY RELEASE")
    log(f"HF repo: {HF_REPO_ID}\n[disk] free: {free_gb():.2f} GB")
    for p in (LLAMA,QUANT):
        if not os.path.isfile(p): raise FileNotFoundError(p)
    r=subprocess.run([LLAMA,"--list-devices"],env=ENV,capture_output=True,text=True,check=False); log((r.stdout or "")+(r.stderr or ""))
    if r.returncode!=0 or "CUDA" not in ((r.stdout or "")+(r.stderr or "")).upper(): raise RuntimeError("CUDA llama.cpp device check failed")
    a=api(); a.create_repo(repo_id=HF_REPO_ID,repo_type="model",exist_ok=True); recover(a)
    f16=ensure_f16(a); s=state()
    if s["stages"].get("F16",{}).get("status")!="PUBLISHED": publish_f16(a,f16)
    else: log("[SKIP] F16 already PUBLISHED")
    s=state()
    if s["stages"].get("Q4_K_M",{}).get("status")!="PUBLISHED":
        if not os.path.isfile(Q4): raise RuntimeError("Validated Q4_K_M artifact is missing locally")
        structural(Q4); publish_quant(a,"Q4_K_M",Q4); os.remove(Q4); log("[CLEAN] Q4_K_M removed locally")
    else: log("[SKIP] Q4_K_M already PUBLISHED")
    for q in REMAINING:
        s=state()
        if s["stages"].get(q,{}).get("status")=="PUBLISHED": log(f"[SKIP] {q} already PUBLISHED"); continue
        out=f"{GGUF_DIR}/lone-wolf-howler-{q.lower()}.gguf"
        try:
            f16=ensure_f16(a); quantize(f16,q,out); mark(q,"VALIDATING",out); smoke(out); mark(q,"VALIDATED",out); publish_quant(a,q,out); os.remove(out); log(f"[DONE] {q} published and cleaned; free={free_gb():.2f} GB")
        except Exception as e:
            mark(q,"FAILED",out,error=f"{type(e).__name__}: {e}"); raise
    if os.path.isfile(F16): os.remove(F16)
    manifest(a); readme(a); runtime_bundle(a); final_verify(api())
    banner("PIPELINE COMPLETE")
    for q in ["F16"]+QUANTS: log(f"GGUF: {q:<8} -> {state()['stages'].get(q,{}).get('status','UNKNOWN')}")
    log(f"HF REPO: https://huggingface.co/{HF_REPO_ID}")
if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: log("Interrupted; published stages are resumable."); raise
    except Exception as e: log(f"\nPIPELINE STOPPED: {type(e).__name__}: {e}\nRerun the same script to resume."); raise
