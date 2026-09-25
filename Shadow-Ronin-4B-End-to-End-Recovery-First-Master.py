#!/usr/bin/env python3
"""
Shadow-Ronin 4B — Recovery-First, Restart-Aware, Self-Healing Pipeline
=======================================================================
Fixes 22.log:
- HF recovery check runs BEFORE any heavy imports.
- Recovery mode uses ONLY huggingface_hub + llama.cpp subprocess.
- bitsandbytes is probed in a subprocess; broken bnb => repair + restart marker.
- bnb is NEVER purged from sys.modules (that was re-registering torch ops).
- No sys.exit(1) inside notebooks => clean _PipelineRestartRequired exit.
"""

# =============================================================================
# PHASE 0: SYNTAX SELF-CHECK
# =============================================================================

import sys
import os


def _syntax_self_check() -> None:
    try:
        path = os.path.abspath(__file__)
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        compile(source, path, "exec")
    except SyntaxError as exc:
        print(
            f"[self-check] SYNTAX ERROR {exc.filename}:{exc.lineno}: {exc.msg}",
            flush=True,
        )
        raise
    except Exception:
        pass


_syntax_self_check()


# =============================================================================
# PHASE 1: STDLIB IMPORTS
# =============================================================================

import gc
import glob
import importlib
import hashlib
import inspect
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import traceback
import types
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")


# =============================================================================
# PHASE 2: CONFIGURATION
# =============================================================================

HF_USER = os.environ.get("HF_USER", "aryansexter")
HF_REPO = os.environ.get("HF_REPO", "Shadow-Ronin-Rag-gguf")
HF_REPO_ID = f"{HF_USER}/{HF_REPO}"

RUNTIME_HF_REPO = os.environ.get("RUNTIME_HF_REPO", "Shadow-Ronin-Runtime")
RUNTIME_HF_REPO_ID = f"{HF_USER}/{RUNTIME_HF_REPO}"

FORCE_SCRATCH_TRAINING = os.environ.get("FORCE_SCRATCH_TRAINING", "0") == "1"
SKIP_RECOVERY = os.environ.get("SKIP_HF_RECOVERY", "0") == "1"
DISABLE_RAG = os.environ.get("DISABLE_RAG", "0") == "1"

KAGGLE_WORKING_DIR = "/kaggle/working"

# Small state that should survive notebook restarts (JSON only) → /kaggle/working
STATE_BASE_DIR = os.path.join(KAGGLE_WORKING_DIR, "persistent_storage")

# Large scratch (FP16, GGUF intermediates, llama.cpp build) → /kaggle/tmp
# Temp space is much larger than the ~20 GB /kaggle/working quota but does
# NOT persist across sessions. HF is the source of truth for large artifacts.
_TMP_CANDIDATES = [
    os.environ.get("SHADOW_RONIN_SCRATCH"),
    "/kaggle/tmp/shadow_ronin_scratch",
    "/tmp/shadow_ronin_scratch",
    os.path.join(KAGGLE_WORKING_DIR, "persistent_storage"),  # fallback
]
SCRATCH_BASE_DIR = next(
    (p for p in _TMP_CANDIDATES if p),
    os.path.join(KAGGLE_WORKING_DIR, "persistent_storage"),
)

# Back-compat alias used throughout the script
PERMANENT_BASE_DIR = SCRATCH_BASE_DIR

PERMANENT_DATASET_DIR = os.path.join(SCRATCH_BASE_DIR, "dataset_cache")
PERMANENT_CHECKPOINT_DIR = os.path.join(SCRATCH_BASE_DIR, "checkpoints")
PERMANENT_GRPO_CHECKPOINT_DIR = os.path.join(SCRATCH_BASE_DIR, "grpo_checkpoints")
PERMANENT_EXPORT_DIR = os.path.join(SCRATCH_BASE_DIR, "gguf_ready_model")
DOWNLOADED_FP16_DIR = os.path.join(SCRATCH_BASE_DIR, "downloaded_fp16_model")
GGUF_OUTPUT_DIR = os.path.join(SCRATCH_BASE_DIR, "gguf_quantized_models")
DOWNLOADED_DATA_DIR = os.path.join(KAGGLE_WORKING_DIR, "extra_data")

RAG_INDEX_DIR = os.path.join(STATE_BASE_DIR, "rag_index")
RAG_DB_PATH = os.path.join(RAG_INDEX_DIR, "rag.sqlite")
RAG_MANIFEST_PATH = os.path.join(RAG_INDEX_DIR, "manifest.json")

NORMALIZED_JSONL_DIR = os.path.join(SCRATCH_BASE_DIR, "normalized_jsonl")
VALIDATION_REPORT_PATH = os.path.join(STATE_BASE_DIR, "jsonl_validation_report.json")
RUNTIME_COMPAT_REPORT_PATH = os.path.join(STATE_BASE_DIR, "runtime_compatibility_report.json")
HF_RECOVERY_REPORT_PATH = os.path.join(STATE_BASE_DIR, "hf_recovery_report.json")
PIPELINE_MARKER_PATH = os.path.join(STATE_BASE_DIR, "pipeline_marker.json")
GGUF_STATE_PATH = os.path.join(STATE_BASE_DIR, "gguf_stage_state.json")

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "aryankumaryalangi")
KAGGLE_DATASET_NAME = os.environ.get("KAGGLE_DATASET_NAME", "Mydataset2")
KAGGLE_DATASET_SLUG = re.sub(r"[^a-z0-9-]+", "-", KAGGLE_DATASET_NAME.lower()).strip("-")
SPECIFIC_KAGGLE_DATASET_PATH = os.path.join(
    "/kaggle/input/datasets", KAGGLE_USERNAME, KAGGLE_DATASET_SLUG,
)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2",
)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_PROMPT = (
    "You are a natural, helpful, and empathetic AI assistant. "
    f"First think through the problem step by step inside {THINK_OPEN}{THINK_CLOSE} tags, "
    f"then give your final answer inside {ANSWER_OPEN}{ANSWER_CLOSE} tags."
)

QUANT_TARGETS = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K", "IQ4_XS"]

GRPO_MAX_EXAMPLES = 48
GRPO_NUM_GENERATIONS = 2
GRPO_MAX_COMPLETION_LENGTH = 64
GRPO_MAX_PROMPT_LENGTH = 320
GRPO_MAX_STEPS = 12
GRPO_BATCH_SIZE = 2
GRPO_GRAD_ACCUM = 4

SFT_MAX_LENGTH = 512
SFT_MAX_STEPS = -1

RAG_TOP_K = 5
RAG_CHUNK_SIZE = 1200
RAG_CHUNK_OVERLAP = 150

GGUF_SMOKE_CTX = 64
GGUF_SMOKE_N_PREDICT = 4
GGUF_SMOKE_THREADS = 4
GGUF_SMOKE_BATCH = 16
GGUF_SMOKE_TIMEOUT_MIN = 30
GGUF_SMOKE_TIMEOUT_PER_GB = 12
GGUF_SMOKE_TIMEOUT_MAX = 120

RUNTIME_COMPAT_FAIL_FAST = True
MAX_INSTALL_RETRIES = 2
MAX_HF_UPLOAD_RETRIES = 3
MAX_HF_DOWNLOAD_RETRIES = 3

RESTART_EXIT_CODE = 75
BNB_REPAIRED_MARKER = "bnb_repaired_restart_required"

PIN_MATRIX = {
    "transformers_5": "transformers>=5.0.0,<5.2.0",
    "transformers_4": "transformers>=4.57.0,<4.58.0",
    "transformers_legacy": "transformers>=4.44.0,<4.58.0",
    "tokenizers_5": "tokenizers>=0.22.0,<0.23.0",
    "tokenizers_4": "tokenizers>=0.20.0,<0.23.0",
    "trl_5": "trl>=1.0.0,<2.0.0",
    "trl_4": "trl>=0.12.0,<2.0.0",
    "peft": "peft>=0.13.0,<1.0.0",
    "hub_1": "huggingface_hub>=1.0.0,<2.0.0",
    "hub_0": "huggingface_hub>=0.30.0,<1.0.0",
    "accelerate": "accelerate>=0.34.0",
    "bitsandbytes": "bitsandbytes>=0.43.0",
    "datasets": "datasets>=3.0.0",
    "sentence_transformers": "sentence-transformers>=3.0.0",
    "sentencepiece": "sentencepiece>=0.2.0",
    "gguf": "gguf>=0.9.0",
    "protobuf": "protobuf>=4.25.0",
    "psutil": "psutil>=5.9.0",
}

for _d in (
    STATE_BASE_DIR, SCRATCH_BASE_DIR,
    PERMANENT_BASE_DIR, PERMANENT_DATASET_DIR, PERMANENT_CHECKPOINT_DIR,
    PERMANENT_GRPO_CHECKPOINT_DIR, PERMANENT_EXPORT_DIR, DOWNLOADED_FP16_DIR,
    GGUF_OUTPUT_DIR, DOWNLOADED_DATA_DIR, RAG_INDEX_DIR, NORMALIZED_JSONL_DIR,
):
    os.makedirs(_d, exist_ok=True)


# =============================================================================
# PHASE 3: LOGGING / HELPERS
# =============================================================================

_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[Shadow-Ronin] {msg}", flush=True)


class _PipelineRestartRequired(Exception):
    """Signal that the kernel must restart to complete bnb recovery."""


def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    check: bool = True,
    capture: bool = False,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(x) for x in cmd))
    return subprocess.run(
        cmd, cwd=cwd, check=check, text=True,
        capture_output=capture, env=env, timeout=timeout,
    )


def retry(
    exceptions: Tuple[type, ...] = (Exception,),
    tries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    label: str = "operation",
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = delay
            last_exc = None
            for attempt in range(1, tries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == tries:
                        break
                    log(f"[retry] {label} {attempt}/{tries} failed "
                        f"({type(exc).__name__}: {exc}); wait {wait:.1f}s")
                    time.sleep(wait)
                    wait *= backoff
            raise last_exc if last_exc else RuntimeError(f"{label} failed")
        return wrapper
    return decorator


@retry(tries=MAX_INSTALL_RETRIES, delay=3.0, label="pip install")
def _pip_install_once(packages: List[str]) -> None:
    run_cmd([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-cache-dir", "--disable-pip-version-check",
    ] + packages)


def pip_install(packages: List[str]) -> bool:
    try:
        _pip_install_once(packages)
        return True
    except Exception as exc:
        log(f"[pip] install failed: {exc}")
        return False


def _pip_uninstall(packages: List[str]) -> None:
    try:
        run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", "-q"] + packages,
                check=False)
    except Exception as exc:
        log(f"[pip] uninstall warning: {exc}")


# =============================================================================
# PHASE 4: RAM / CLEANUP
# =============================================================================

def _free_ram_mb() -> int:
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def _log_ram(prefix: str = "") -> None:
    free = _free_ram_mb()
    if free >= 0:
        log(f"{prefix}Free RAM: {free} MB")


def _kill_lingering_llama_cli() -> None:
    for pat in ("llama-cli", "llama.cpp/build/bin", "llama-quantize"):
        try:
            subprocess.run(["pkill", "-9", "-f", pat],
                           check=False, capture_output=True, timeout=10)
        except Exception:
            pass


def _deep_cleanup(label: str = "") -> None:
    if label:
        log(f"[cleanup] {label}")
    _kill_lingering_llama_cli()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
    _log_ram("[cleanup] ")


# =============================================================================
# PHASE 5: RUNTIME COMPAT REPORT
# =============================================================================

_runtime_compat_report: Dict[str, Any] = {
    "generated_at": None, "status": "unknown",
    "platform": {}, "torch": {}, "cuda": {},
    "bnb_probe": {}, "restart_events": [],
    "transformers": {}, "trl": {}, "peft": {}, "tokenizers": {},
    "huggingface_hub": {}, "torchvision": {}, "bitsandbytes": {},
    "sentence_transformers": {}, "datasets": {},
    "tokenizer_chat_template": {}, "model_forward_preflight": {},
    "sft_compat": {}, "grpo_compat": {},
    "warnings": [], "errors": [],
    "packages_installed": [], "packages_failed": [],
}


def _compat_record(section: str, key: str, value: Any) -> None:
    try:
        _runtime_compat_report.setdefault(section, {})[key] = value
    except Exception:
        pass


def _compat_warn(message: str) -> None:
    try:
        _runtime_compat_report["warnings"].append(message)
    except Exception:
        pass
    log(f"[runtime-compat][WARN] {message}")


def _compat_error(message: str) -> None:
    try:
        _runtime_compat_report["errors"].append(message)
    except Exception:
        pass
    log(f"[runtime-compat][ERROR] {message}")


def _write_runtime_compat_report() -> None:
    _runtime_compat_report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if _runtime_compat_report["errors"]:
        _runtime_compat_report["status"] = "failed"
    elif _runtime_compat_report["warnings"]:
        _runtime_compat_report["status"] = "ok_with_warnings"
    else:
        _runtime_compat_report["status"] = "ok"
    try:
        with open(RUNTIME_COMPAT_REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(_runtime_compat_report, fh, indent=2, ensure_ascii=False)
        log(f"[runtime-compat] report written: {RUNTIME_COMPAT_REPORT_PATH} "
            f"(status={_runtime_compat_report['status']})")
    except Exception as exc:
        log(f"[runtime-compat] report write failed: {exc}")


# =============================================================================
# PHASE 6: MARKER MANAGEMENT
# =============================================================================

def _read_marker() -> Optional[Dict[str, Any]]:
    if not os.path.isfile(PIPELINE_MARKER_PATH):
        return None
    try:
        with open(PIPELINE_MARKER_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        log(f"[marker] read failed: {exc}")
        return None


def _write_marker(data: Dict[str, Any]) -> None:
    data = dict(data)
    data["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = PIPELINE_MARKER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, PIPELINE_MARKER_PATH)


def _clear_marker() -> None:
    try:
        if os.path.isfile(PIPELINE_MARKER_PATH):
            os.remove(PIPELINE_MARKER_PATH)
    except Exception:
        pass


# =============================================================================
# PHASE 7: HARDWARE DETECTION (torch only, never bnb)
# =============================================================================

def _detect_hardware() -> Dict[str, Any]:
    import platform
    try:
        _compat_record("platform", "python", sys.version.split()[0])
        _compat_record("platform", "executable", sys.executable)
        _compat_record("platform", "system", platform.system())
        _compat_record("platform", "release", platform.release())
        _compat_record("platform", "machine", platform.machine())
    except Exception as exc:
        _compat_warn(f"platform detection issue: {exc}")

    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["device_count"] = (
            torch.cuda.device_count() if info["cuda_available"] else 0
        )
        if info["cuda_available"]:
            try:
                info["device_name_0"] = torch.cuda.get_device_name(0)
                info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                pass
        for k, v in info.items():
            _compat_record("torch", k, v)
    except Exception as exc:
        _compat_error(f"torch import failed: {exc}")
        info["torch_error"] = str(exc)

    log(f"[hardware] torch={info.get('torch_version')} "
        f"cuda={info.get('cuda_available')} "
        f"cuda_version={info.get('cuda_version')} "
        f"devices={info.get('device_count')}")
    return info


# =============================================================================
# PHASE 8: HF AUTH (huggingface_hub only)
# =============================================================================

def get_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(name)
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
            try:
                v = client.get_secret(name)
                if v:
                    return v
            except Exception:
                continue
    except Exception:
        pass
    return None


def authenticate_hf() -> None:
    token = get_hf_token()
    if not token:
        log("[HF] token not found; continuing with anonymous access")
        return
    try:
        from huggingface_hub import login
        try:
            login(token=token, add_to_git_credential=False)
        except TypeError:
            login(token=token)
        log("[HF] authentication completed")
    except Exception as exc:
        log(f"[HF] login warning: {exc}")


def _ensure_huggingface_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        log("[HF] huggingface_hub missing; installing minimal version")
        return pip_install(["huggingface_hub>=0.30.0"])


# =============================================================================
# PHASE 9: HF RECOVERY CHECK (before any heavy imports)
# =============================================================================

def check_hf_recovery() -> Dict[str, Any]:
    result = {
        "available": False, "fp16_available": False,
        "gguf_published": [], "validation_available": [],
        "files": [], "probe_error": None,
    }
    if not _ensure_huggingface_hub():
        result["probe_error"] = "huggingface_hub unavailable"
        return result
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        result["probe_error"] = f"HfApi import failed: {exc}"
        return result

    token = get_hf_token()
    try:
        api = HfApi(token=token)
    except Exception as exc:
        result["probe_error"] = f"HfApi init failed: {exc}"
        return result

    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type="model")
    except Exception as exc:
        result["probe_error"] = f"repo not accessible: {exc}"
        log(f"[HF recovery] repo not accessible ({exc})")
        return result

    result["available"] = True
    try:
        files = sorted(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model"))
    except Exception as exc:
        result["probe_error"] = f"list_repo_files failed: {exc}"
        return result
    result["files"] = files

    fp16_required = ["fp16_model/config.json", "fp16_model/tokenizer_config.json"]
    fp16_gguf = any(
        f in files for f in (
            "fp16_model/gguf_output/shadow-ronin-4b-f16.gguf",
            "gguf/shadow-ronin-4b-f16.gguf",
        )
    )
    if all(f in files for f in fp16_required) or fp16_gguf:
        result["fp16_available"] = True
    for f in files:
        if f.startswith("gguf/") and f.endswith(".gguf"):
            result["gguf_published"].append(os.path.basename(f))
        if f.startswith("validation/"):
            result["validation_available"].append(f)

    try:
        with open(HF_RECOVERY_REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return result


@retry(tries=MAX_HF_DOWNLOAD_RETRIES, delay=5.0, label="HF snapshot_download")
def _hf_snapshot(repo_id: str, local_dir: str, token: Optional[str],
                 allow_patterns: Optional[List[str]] = None) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token,
                      allow_patterns=allow_patterns)


def _local_fp16_looks_valid(dest_dir: str) -> bool:
    """Cheap structural check — enough to skip a multi-GB re-download."""
    if not os.path.isdir(dest_dir):
        return False
    for req in ("config.json", "tokenizer_config.json"):
        if not os.path.isfile(os.path.join(dest_dir, req)):
            return False
    safes = [f for f in os.listdir(dest_dir) if f.endswith(".safetensors")]
    return len(safes) >= 1


def _free_disk_mb(path: str = PERMANENT_BASE_DIR) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024 * 1024)
    except Exception:
        return -1.0


def download_fp16_from_hf(dest_dir: str) -> str:
    """Prefer an already-validated local FP16 copy; only hit the network if needed."""
    if _local_fp16_looks_valid(dest_dir):
        try:
            validate_recovered_fp16_bnb_free(dest_dir)
            log(f"Reusing existing local FP16 (skip download): {dest_dir}")
            return dest_dir
        except Exception as exc:
            log(f"Local FP16 present but failed validation ({exc}); re-downloading")

    free = _free_disk_mb()
    log(f"[disk] free space before FP16 download: {free:.0f} MB")
    if 0 <= free < 12000:
        raise RuntimeError(
            f"Not enough free disk ({free:.0f} MB) to download FP16 (~8 GB). "
            "Clean /kaggle/working/persistent_storage (build object trees, "
            ".cache, old GGUF intermediates) and retry."
        )

    token = get_hf_token()
    os.makedirs(dest_dir, exist_ok=True)
    log(f"Downloading fp16_model/ from HF: {HF_REPO_ID}")
    _hf_snapshot(HF_REPO_ID, dest_dir, token, ["fp16_model/*"])

    nested = os.path.join(dest_dir, "fp16_model")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(dest_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        try:
            os.rmdir(nested)
        except OSError:
            pass

    # Drop HF download cache to reclaim space immediately
    cache = os.path.join(dest_dir, ".cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache, ignore_errors=True)
        log("[disk] removed HF download cache under downloaded_fp16_model")
    return dest_dir


def validate_recovered_fp16_bnb_free(fp16_dir: str) -> None:
    """bnb/transformers-free structural validation of the recovered FP16 model."""
    log("Validating recovered FP16 model (bnb-free structural check)...")

    for f in ["config.json", "tokenizer_config.json"]:
        if not os.path.isfile(os.path.join(fp16_dir, f)):
            raise RuntimeError(f"Recovered FP16 missing {f}")
    log("  config.json .......... OK")
    log("  tokenizer_config.json  OK")

    with open(os.path.join(fp16_dir, "config.json"), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if "model_type" not in cfg:
        raise RuntimeError("config.json missing model_type")
    log(f"  model_type ........... {cfg['model_type']}")

    safes = [f for f in os.listdir(fp16_dir) if f.endswith(".safetensors")]
    if not safes:
        raise RuntimeError("no .safetensors files present")
    log(f"  safetensors .......... {len(safes)} shard(s)")

    idx = os.path.join(fp16_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        shards = set(d.get("weight_map", {}).values())
        missing = [s for s in shards if not os.path.isfile(os.path.join(fp16_dir, s))]
        if missing:
            raise RuntimeError(f"missing shards: {missing}")
        log(f"  index shards ......... {len(shards)} present")

    tc_path = os.path.join(fp16_dir, "tokenizer_config.json")
    with open(tc_path, "r", encoding="utf-8") as fh:
        tc = json.load(fh)
    has_template = bool(tc.get("chat_template"))
    has_jinja = os.path.isfile(os.path.join(fp16_dir, "chat_template.jinja"))
    if not (has_template or has_jinja):
        raise RuntimeError("no chat_template found (embedded or .jinja)")
    log(f"  chat_template ........ OK (embedded={has_template} jinja={has_jinja})")

    log("HF FP16 validation (bnb-free): PASS")


def recover_validation_artifacts(files: List[str]) -> None:
    token = get_hf_token()
    mapping = {
        "validation/jsonl_validation_report.json": VALIDATION_REPORT_PATH,
        "validation/manifest.json": RAG_MANIFEST_PATH,
        "validation/rag.sqlite": RAG_DB_PATH,
        "validation/runtime_compatibility_report.json": RUNTIME_COMPAT_REPORT_PATH,
    }
    to_fetch = [r for r in mapping if r in files]
    if not to_fetch:
        return
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _hf_snapshot(HF_REPO_ID, tmp, token, to_fetch)
            for remote, local in mapping.items():
                src = os.path.join(tmp, remote)
                if os.path.isfile(src):
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    shutil.copy2(src, local)
                    log(f"Recovered: {remote} -> {local}")
    except Exception as exc:
        log(f"Validation artifact recovery skipped ({exc})")


# =============================================================================
# PHASE 10: GGUF STATE
# =============================================================================

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_gguf_state() -> Dict[str, Any]:
    if not os.path.isfile(GGUF_STATE_PATH):
        return {"version": 1, "stages": {}}
    try:
        with open(GGUF_STATE_PATH, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            d.setdefault("version", 1)
            d.setdefault("stages", {})
            return d
    except Exception:
        pass
    return {"version": 1, "stages": {}}


def _save_gguf_state(state: Dict[str, Any]) -> None:
    tmp = GGUF_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, GGUF_STATE_PATH)


def _mark_gguf_stage(quant: str, status: str,
                     artifact: Optional[str] = None,
                     error: Optional[str] = None) -> None:
    state = _load_gguf_state()
    item = {"status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if artifact:
        item["artifact"] = os.path.abspath(artifact)
        if os.path.isfile(artifact):
            item["size_bytes"] = os.path.getsize(artifact)
    if error:
        item["error"] = error
    state["stages"][quant] = item
    _save_gguf_state(state)


def recover_gguf_stage_state(published_gguf_names: List[str]) -> None:
    state = _load_gguf_state()
    mapping = {f"shadow-ronin-4b-{q.lower()}.gguf": q for q in QUANT_TARGETS}
    mapping["shadow-ronin-4b-f16.gguf"] = "F16"
    updated = 0
    for fname in published_gguf_names:
        if fname in mapping:
            quant = mapping[fname]
            state["stages"][quant] = {
                "status": "PUBLISHED",
                "artifact": f"hf://{HF_REPO_ID}/gguf/{fname}",
                "recovered_from_hf": True,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            updated += 1
    if updated:
        _save_gguf_state(state)
        log(f"[GGUF] pre-populated state: {updated} recovered stage(s)")


# =============================================================================
# PHASE 11: LLAMA.CPP + GGUF (no Python ML deps)
# =============================================================================

def _nvidia_gpu_info() -> Dict[str, Any]:
    """Detect an NVIDIA GPU without importing torch."""
    info: Dict[str, Any] = {"available": False, "names": [], "raw": ""}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False, text=True, capture_output=True, timeout=20,
        )
        raw = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        info["raw"] = raw[-4000:]
        names = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
        info["names"] = names
        info["available"] = bool(r.returncode == 0 and names)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _cuda_arch_for_nvidia_names(names: List[str]) -> Optional[str]:
    """Return a compact CMAKE_CUDA_ARCHITECTURES value for common GPUs.

    Unknown GPUs deliberately return None so llama.cpp's non-native CUDA
    architecture selection remains responsible for portability.
    """
    joined = " ".join(names).lower()
    known = [
        (("tesla t4", "t4"), "75"),
        (("tesla p100", "p100"), "60"),
        (("tesla v100", "v100"), "70"),
        (("a100",), "80"),
        (("a10",), "86"),
        (("a40",), "86"),
        (("l4",), "89"),
        (("l40", "l40s"), "89"),
        (("rtx 30", "rtx 3090", "rtx 3080", "rtx 3070", "rtx 3060"), "86"),
        (("rtx 40", "rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060"), "89"),
        (("h100", "h200"), "90"),
    ]
    for needles, arch in known:
        if any(n in joined for n in needles):
            return arch
    return None


def _find_cuda_home() -> Optional[str]:
    """Locate a usable CUDA toolkit root (nvcc + include/lib)."""
    candidates: List[str] = []
    for env_key in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        v = os.environ.get(env_key)
        if v:
            candidates.append(v)
    try:
        r = subprocess.run(
            ["which", "nvcc"], check=False, text=True,
            capture_output=True, timeout=10,
        )
        nvcc = (r.stdout or "").strip()
        if nvcc and os.path.isfile(nvcc):
            home = os.path.dirname(os.path.dirname(os.path.abspath(nvcc)))
            candidates.append(home)
    except Exception:
        pass
    candidates.extend([
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-11.8",
        "/opt/cuda",
    ])
    seen = set()
    for c in candidates:
        c = os.path.abspath(c) if c else ""
        if not c or c in seen:
            continue
        seen.add(c)
        nvcc = os.path.join(c, "bin", "nvcc")
        if os.path.isfile(nvcc):
            return c
    return None


def _find_libcuda() -> Dict[str, Any]:
    """Locate libcuda.so (prefer real driver, then toolkit stubs).

    Kaggle / container hosts often ship the CUDA toolkit but CMake's
    CUDAToolkit package does not always create the CUDA::cuda_driver imported
    target. Providing an explicit path to libcuda.so (or the stubs library)
    lets us inject the missing target.
    """
    result: Dict[str, Any] = {
        "path": None, "dir": None, "is_stub": False, "candidates": [],
    }
    search_dirs: List[str] = []
    search_dirs.extend([
        "/usr/local/nvidia/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/lib/x86_64-linux-gnu",
        "/lib64",
    ])
    cuda_home = _find_cuda_home()
    if cuda_home:
        search_dirs.extend([
            os.path.join(cuda_home, "lib64", "stubs"),
            os.path.join(cuda_home, "lib", "stubs"),
            os.path.join(cuda_home, "targets", "x86_64-linux", "lib", "stubs"),
            os.path.join(cuda_home, "lib64"),
            os.path.join(cuda_home, "lib"),
        ])
    try:
        r = subprocess.run(
            ["ldconfig", "-p"], check=False, text=True,
            capture_output=True, timeout=15,
        )
        for line in (r.stdout or "").splitlines():
            if "libcuda.so" in line:
                parts = line.strip().split("=>")
                if len(parts) == 2:
                    p = parts[1].strip()
                    if os.path.isfile(p):
                        search_dirs.append(os.path.dirname(p))
                        result["candidates"].append(p)
    except Exception:
        pass
    for part in (os.environ.get("LD_LIBRARY_PATH") or "").split(":"):
        if part:
            search_dirs.append(part)

    seen_dirs = set()
    real_hit: Optional[str] = None
    stub_hit: Optional[str] = None
    for d in search_dirs:
        d = os.path.abspath(d) if d else ""
        if not d or d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(d)
        for name in ("libcuda.so", "libcuda.so.1"):
            p = os.path.join(d, name)
            if not os.path.isfile(p) and not os.path.islink(p):
                if not os.path.lexists(p):
                    continue
            result["candidates"].append(p)
            is_stub = "stubs" in d.split(os.sep)
            if is_stub:
                if stub_hit is None:
                    stub_hit = p
            else:
                if real_hit is None:
                    real_hit = p

    chosen = real_hit or stub_hit
    if chosen:
        result["path"] = chosen
        result["dir"] = os.path.dirname(chosen)
        result["is_stub"] = "stubs" in result["dir"].split(os.sep)
    return result


def _write_cuda_driver_cmake_workaround(path: str, libcuda_path: str) -> None:
    """Write a CMake fragment that defines CUDA::cuda_driver when missing."""
    safe = libcuda_path.replace("\\", "/")
    content = f"""# Auto-generated by Shadow-Ronin — CUDA::cuda_driver workaround
if(NOT TARGET CUDA::cuda_driver)
  if(NOT EXISTS "{safe}")
    message(FATAL_ERROR
      "Shadow-Ronin CUDA workaround: libcuda not found at {safe}")
  endif()
  add_library(CUDA::cuda_driver SHARED IMPORTED GLOBAL)
  set_target_properties(CUDA::cuda_driver PROPERTIES
    IMPORTED_LOCATION "{safe}"
  )
  message(STATUS
    "Shadow-Ronin: defined CUDA::cuda_driver -> {safe}")
endif()
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _detect_cuda_toolkit() -> Dict[str, Any]:
    """Collect CUDA toolkit + driver facts for logging and CMake."""
    info: Dict[str, Any] = {
        "cuda_home": None,
        "nvcc": None,
        "nvcc_version": None,
        "libcuda": None,
        "libcuda_dir": None,
        "libcuda_is_stub": False,
    }
    home = _find_cuda_home()
    info["cuda_home"] = home
    if home:
        nvcc = os.path.join(home, "bin", "nvcc")
        if os.path.isfile(nvcc):
            info["nvcc"] = nvcc
            try:
                r = subprocess.run(
                    [nvcc, "--version"], check=False, text=True,
                    capture_output=True, timeout=15,
                )
                text = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
                m = re.search(r"release\s+([\d.]+)", text, re.IGNORECASE)
                if m:
                    info["nvcc_version"] = m.group(1)
                else:
                    info["nvcc_version"] = text[-200:]
            except Exception as exc:
                info["nvcc_version"] = f"probe failed: {exc}"

    lib = _find_libcuda()
    info["libcuda"] = lib.get("path")
    info["libcuda_dir"] = lib.get("dir")
    info["libcuda_is_stub"] = bool(lib.get("is_stub"))
    info["libcuda_candidates"] = lib.get("candidates", [])[:12]
    return info


def _llama_cli_cuda_devices(cli: str) -> Tuple[bool, str]:
    """Verify that this llama-cli binary can enumerate CUDA devices."""
    try:
        r = subprocess.run(
            [cli, "--list-devices"], check=False, text=True,
            capture_output=True, timeout=30,
        )
        text = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        low = text.lower()
        cuda = (
            r.returncode == 0 and
            ("cuda" in low or "nvidia" in low) and
            not any(x in low for x in ("cuda backend not found", "no cuda", "not compiled with cuda"))
        )
        return cuda, text[-6000:]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _verify_llama_cuda_backend(cli: str, required: bool = True) -> bool:
    ok, detail = _llama_cli_cuda_devices(cli)
    _compat_record("llama_cpp", "cuda_backend_ok", ok)
    _compat_record("llama_cpp", "cuda_device_probe", detail)
    if ok:
        log("[llama.cpp] CUDA backend VERIFIED")
        log("[llama.cpp] CUDA devices:\n" + detail)
        return True
    if required:
        raise RuntimeError(
            "llama.cpp was built, but its CUDA backend could not be verified. "
            "A CUDA-capable llama.cpp binary is mandatory when an NVIDIA GPU is "
            f"available. Device probe:\n{detail}"
        )
    log("[llama.cpp] CUDA backend not detected; CPU backend is being used.")
    return False


def _prune_llama_build_objects(build_dir: str) -> None:
    """Remove CMake intermediate object trees; keep bin/ executables.

    On Kaggle the full CUDA build can consume several GB of .o / CMakeFiles.
    After a successful link we only need the binaries under bin/.
    """
    if not os.path.isdir(build_dir):
        return
    bin_dir = os.path.join(build_dir, "bin")
    kept = []
    if os.path.isdir(bin_dir):
        for name in os.listdir(bin_dir):
            p = os.path.join(bin_dir, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                kept.append(name)
    # Remove heavy intermediate dirs; leave bin/, CMakeCache, workaround cmake
    for entry in os.listdir(build_dir):
        if entry in ("bin", "CMakeCache.txt", "shadow_ronin_cuda_driver.cmake",
                     "CMakeFiles"):  # CMakeFiles is large but needed for rebuild skip
            # Drop CMakeFiles object trees selectively
            if entry == "CMakeFiles":
                cmf = os.path.join(build_dir, "CMakeFiles")
                for sub in os.listdir(cmf):
                    subp = os.path.join(cmf, sub)
                    if os.path.isdir(subp) and sub not in ("CMakeTmp",):
                        # Keep only the stamp/cache bits; wipe object dirs
                        for root, dirs, files in os.walk(subp):
                            for f in files:
                                if f.endswith((".o", ".cu.o", ".cpp.o", ".c.o")):
                                    try:
                                        os.remove(os.path.join(root, f))
                                    except OSError:
                                        pass
            continue
        path = os.path.join(build_dir, entry)
        try:
            if os.path.isdir(path):
                # Skip bin; remove vendor/ggml/src object trees etc.
                if entry in ("ggml", "src", "common", "tools", "examples",
                             "vendor", "tests", "pocs", "app"):
                    shutil.rmtree(path, ignore_errors=True)
            elif entry.endswith((".o", ".a")) and entry not in kept:
                os.remove(path)
        except OSError:
            pass
    free = _free_disk_mb()
    log(f"[llama.cpp] pruned build intermediates under {build_dir}; "
        f"kept binaries: {kept}; free disk now ~{free:.0f} MB")


def ensure_llama_cpp() -> str:
    """Build llama.cpp with CUDA when an NVIDIA GPU is available.

    IMPORTANT: never reuse the old CPU-only build for a CUDA run. CUDA and CPU
    builds use separate directories (build_cuda / build_cpu).

    Kaggle-specific: the CUDA toolkit is present and nvcc works, but CMake's
    CUDAToolkit package often fails to create the CUDA::cuda_driver imported
    target that ggml-cuda links against. We detect libcuda.so (real or stubs)
    and inject a small CMake fragment that defines the missing target.
    """
    repo_dir = os.path.join(PERMANENT_BASE_DIR, "llama.cpp")
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        run_cmd(["git", "clone", "--depth", "1",
                 "https://github.com/ggml-org/llama.cpp.git", repo_dir])
    else:
        try:
            run_cmd(["git", "fetch", "--depth", "1", "origin", "master"],
                    cwd=repo_dir, check=False)
            run_cmd(["git", "reset", "--hard", "origin/master"],
                    cwd=repo_dir, check=False)
        except Exception as exc:
            log(f"[llama.cpp] update warning: {exc}")

    nvidia = _nvidia_gpu_info()
    cuda_requested = bool(nvidia.get("available"))
    _compat_record("llama_cpp", "nvidia_gpu_available", cuda_requested)
    _compat_record("llama_cpp", "nvidia_gpu_names", nvidia.get("names", []))

    if cuda_requested:
        toolkit = _detect_cuda_toolkit()
        _compat_record("llama_cpp", "cuda_home", toolkit.get("cuda_home"))
        _compat_record("llama_cpp", "nvcc", toolkit.get("nvcc"))
        _compat_record("llama_cpp", "nvcc_version", toolkit.get("nvcc_version"))
        _compat_record("llama_cpp", "libcuda", toolkit.get("libcuda"))
        _compat_record("llama_cpp", "libcuda_is_stub", toolkit.get("libcuda_is_stub"))

        log(f"[llama.cpp] CUDA home ...... {toolkit.get('cuda_home')}")
        log(f"[llama.cpp] nvcc ........... {toolkit.get('nvcc')} "
            f"(version={toolkit.get('nvcc_version')})")
        log(f"[llama.cpp] libcuda ........ {toolkit.get('libcuda')} "
            f"(stub={toolkit.get('libcuda_is_stub')})")

        if not toolkit.get("cuda_home") or not toolkit.get("nvcc"):
            raise RuntimeError(
                "NVIDIA GPU is present but no CUDA toolkit (nvcc) was found. "
                "Cannot build a CUDA llama.cpp backend. "
                f"Toolkit probe: {toolkit}"
            )
        if not toolkit.get("libcuda"):
            raise RuntimeError(
                "NVIDIA GPU is present and CUDA toolkit was found, but "
                "libcuda.so could not be located (neither the real driver nor "
                "the toolkit stubs library). CMake cannot create "
                "CUDA::cuda_driver without it. "
                f"Candidates seen: {toolkit.get('libcuda_candidates')}"
            )

        build_dir = os.path.join(repo_dir, "build_cuda")
        cli = os.path.join(build_dir, "bin", "llama-cli")
        quant_bin = os.path.join(build_dir, "bin", "llama-quantize")
        quant_alt = os.path.join(build_dir, "bin", "quantize")
        arch = _cuda_arch_for_nvidia_names(nvidia.get("names", []))
        cache = os.path.join(build_dir, "CMakeCache.txt")
        workaround = os.path.join(build_dir, "shadow_ronin_cuda_driver.cmake")

        quant_ok = os.path.isfile(quant_bin) or os.path.isfile(quant_alt)
        need_configure = not (os.path.isfile(cli) and quant_ok)
        if os.path.isfile(cache):
            cache_text = Path(cache).read_text(errors="ignore")
            cuda_on = bool(re.search(r"GGML_CUDA:BOOL=ON", cache_text))
            native_off = bool(re.search(r"GGML_NATIVE:BOOL=OFF", cache_text))
            driver_ok = os.path.isfile(workaround)
            if not (cuda_on and native_off and driver_ok and quant_ok):
                need_configure = True
                log("[llama.cpp] existing CUDA build incomplete / missing "
                    "quantizer or driver workaround; will reconfigure")

        if need_configure:
            if os.path.isdir(build_dir):
                shutil.rmtree(build_dir, ignore_errors=True)
            os.makedirs(build_dir, exist_ok=True)

            _write_cuda_driver_cmake_workaround(
                workaround, toolkit["libcuda"],
            )
            log(f"[llama.cpp] wrote CUDA::cuda_driver workaround -> {workaround}")

            cmake_args = [
                "cmake", "-S", repo_dir, "-B", build_dir,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_CUDA=ON",
                "-DGGML_NATIVE=OFF",
                "-DBUILD_SHARED_LIBS=OFF",
                f"-DCUDAToolkit_ROOT={toolkit['cuda_home']}",
                f"-DCMAKE_CUDA_COMPILER={toolkit['nvcc']}",
                f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={workaround}",
            ]
            lib_paths = []
            if toolkit.get("libcuda_dir"):
                lib_paths.append(toolkit["libcuda_dir"])
            stubs = os.path.join(toolkit["cuda_home"], "lib64", "stubs")
            if os.path.isdir(stubs) and stubs not in lib_paths:
                lib_paths.append(stubs)
            lib64 = os.path.join(toolkit["cuda_home"], "lib64")
            if os.path.isdir(lib64) and lib64 not in lib_paths:
                lib_paths.append(lib64)
            if lib_paths:
                cmake_args.append(
                    "-DCMAKE_LIBRARY_PATH=" + ";".join(lib_paths)
                )

            if arch:
                cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
                log(f"[llama.cpp] detected NVIDIA GPU {nvidia['names']}; "
                    f"CUDA arch={arch}")
            else:
                log("[llama.cpp] NVIDIA GPU detected; using llama.cpp "
                    "non-native CUDA architectures")

            env = os.environ.copy()
            env["CUDA_HOME"] = toolkit["cuda_home"]
            env["CUDA_PATH"] = toolkit["cuda_home"]
            env["CUDACXX"] = toolkit["nvcc"]
            stub_dir = os.path.join(toolkit["cuda_home"], "lib64", "stubs")
            path_prefix = []
            if os.path.isdir(stub_dir):
                path_prefix.append(stub_dir)
            if toolkit.get("libcuda_dir"):
                path_prefix.append(toolkit["libcuda_dir"])
            if path_prefix:
                existing = env.get("LIBRARY_PATH", "")
                env["LIBRARY_PATH"] = ":".join(
                    path_prefix + ([existing] if existing else [])
                )
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = ":".join(
                    ([toolkit["libcuda_dir"]] if toolkit.get("libcuda_dir") else [])
                    + ([existing_ld] if existing_ld else [])
                )

            log("[llama.cpp] configuring CUDA build with driver workaround")
            run_cmd(cmake_args, env=env)
            run_cmd(
                ["cmake", "--build", build_dir, "--config", "Release", "-j", "4"],
                env=env,
            )
            # Prune CMake object trees to free several GB; keep bin/ binaries.
            _prune_llama_build_objects(build_dir)

        if not os.path.isfile(cli):
            raise FileNotFoundError("CUDA llama-cli was not produced")
        # Also ensure quantizer binary is present (same bin dir)
        quant = os.path.join(build_dir, "bin", "llama-quantize")
        if not os.path.isfile(quant):
            quant_alt = os.path.join(build_dir, "bin", "quantize")
            if not os.path.isfile(quant_alt):
                raise FileNotFoundError(
                    f"CUDA build produced llama-cli but not llama-quantize "
                    f"(checked {quant} and {quant_alt})"
                )
        _verify_llama_cuda_backend(cli, required=True)
        return cli

    # No NVIDIA GPU: CPU build is valid, but it is explicitly labeled CPU.
    build_dir = os.path.join(repo_dir, "build_cpu")
    cli = os.path.join(build_dir, "bin", "llama-cli")
    if not os.path.isfile(cli):
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)
        run_cmd([
            "cmake", "-S", repo_dir, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
            "-DGGML_CUDA=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
        ])
        run_cmd(["cmake", "--build", build_dir, "--config", "Release", "-j", "4"])
    if not os.path.isfile(cli):
        raise FileNotFoundError("CPU llama-cli was not produced")
    log("[llama.cpp] no NVIDIA GPU detected; using CPU backend")
    return cli


def find_llama_convert(repo_dir: str) -> str:
    p = os.path.join(repo_dir, "convert_hf_to_gguf.py")
    if os.path.exists(p):
        return p
    raise FileNotFoundError("convert_hf_to_gguf.py not found")


def find_quantizer(repo_dir: str) -> str:
    """Locate llama-quantize from the CUDA or CPU build tree.

    CUDA builds live under build_cuda/; CPU under build_cpu/. Older layouts
    used a plain build/ directory — still checked for compatibility.
    """
    candidates = [
        os.path.join(repo_dir, "build_cuda", "bin", "llama-quantize"),
        os.path.join(repo_dir, "build_cuda", "bin", "quantize"),
        os.path.join(repo_dir, "build_cpu", "bin", "llama-quantize"),
        os.path.join(repo_dir, "build_cpu", "bin", "quantize"),
        os.path.join(repo_dir, "build", "bin", "llama-quantize"),
        os.path.join(repo_dir, "build", "bin", "quantize"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            log(f"[llama.cpp] quantizer: {p}")
            return p
    raise FileNotFoundError(
        "llama-quantize not found under build_cuda/, build_cpu/, or build/. "
        "Ensure llama.cpp was built successfully."
    )


def _gguf_structural_check(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"GGUF missing: {path}")
    size = os.path.getsize(path)
    if size < 1024 * 1024:
        raise RuntimeError(f"GGUF too small: {path}")
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"Invalid GGUF magic: {path}")
    return {"path": os.path.abspath(path), "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "sha256": _sha256_file(path)}


def _llama_cli_gpu_args(cli: str) -> List[str]:
    """Return GPU offload arguments only after verifying CUDA support."""
    nvidia = _nvidia_gpu_info()
    if not nvidia.get("available"):
        return []
    ok, detail = _llama_cli_cuda_devices(cli)
    if not ok:
        raise RuntimeError(
            "NVIDIA GPU is available but llama-cli has no verified CUDA backend. "
            f"Refusing CPU fallback. Device probe:\n{detail}"
        )
    return ["-ngl", "99"]


def smoke_test_gguf(cli: str, gguf_path: str, query: str) -> None:
    info = _gguf_structural_check(gguf_path)
    size_gb = info["size_bytes"] / (1024 ** 3)
    nvidia = _nvidia_gpu_info()
    gpu = _llama_cli_gpu_args(cli)

    # GPU smoke is deliberately bounded but generous enough for the first CUDA
    # model load. For an 8 GB F16 Qwen-4B GGUF on a Kaggle T4, 180 s is ample;
    # most of the time is spent loading the model, not generating 4 tokens.
    if nvidia.get("available"):
        timeout = min(180, max(60, int(size_gb * 22)))
        mode = "CUDA"
    else:
        timeout = min(GGUF_SMOKE_TIMEOUT_MAX, max(GGUF_SMOKE_TIMEOUT_MIN,
                      int(size_gb * GGUF_SMOKE_TIMEOUT_PER_GB)))
        mode = "CPU"

    log(f"[GGUF smoke] {os.path.basename(gguf_path)} ({info['size_mb']:.1f} MB, "
        f"timeout={timeout}s, backend={mode}, args={gpu or 'none'})")

    cmd = [
        cli, "-m", gguf_path, "-p", query,
        "-n", str(GGUF_SMOKE_N_PREDICT),
        "-c", str(GGUF_SMOKE_CTX),
        "-b", str(GGUF_SMOKE_BATCH),
        "-t", str(GGUF_SMOKE_THREADS),
        *gpu,
        "--no-warmup", "--no-display-prompt",
    ]
    try:
        result = run_cmd(cmd, check=False, capture=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_lingering_llama_cli()
        raise RuntimeError(
            f"GGUF smoke timed out after {timeout}s ({mode} backend): {gguf_path}"
        )

    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode != 0:
        _kill_lingering_llama_cli()
        raise RuntimeError(f"GGUF smoke failed: {combined[-4000:]}")

    if not (result.stdout or result.stderr or "").strip():
        raise RuntimeError(f"GGUF smoke produced no process output: {gguf_path}")

    if nvidia.get("available"):
        low = combined.lower()
        cuda_runtime_seen = any(x in low for x in (
            "cuda", "ggml_cuda", "offloaded", "nvidia", "gpu layers"
        ))
        if not cuda_runtime_seen:
            _kill_lingering_llama_cli()
            raise RuntimeError(
                "GGUF smoke returned successfully but did not expose evidence of "
                f"CUDA/GPU execution. Refusing to mark {gguf_path} as validated. "
                f"Output:\n{combined[-4000:]}"
            )

    log("[GGUF smoke] PASS")
    log(f"[GGUF smoke] runtime output tail:\n{combined[-1200:]}")
    _kill_lingering_llama_cli()
    _deep_cleanup(f"after GGUF smoke {os.path.basename(gguf_path)}")


def _ensure_f16_gguf(fp16_dir: str, converter: str) -> str:
    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    f16 = os.path.join(GGUF_OUTPUT_DIR, "shadow-ronin-4b-f16.gguf")
    if os.path.isfile(f16):
        try:
            _gguf_structural_check(f16)
            log("[GGUF] reusing existing local F16")
            return f16
        except Exception:
            try:
                os.remove(f16)
            except Exception:
                pass
    log("[GGUF] converting FP16 -> F16")
    free = _free_disk_mb()
    log(f"[disk] free before F16 convert: {free:.0f} MB")
    if 0 <= free < 10000:
        raise RuntimeError(
            f"Not enough free disk ({free:.0f} MB) for F16 convert. "
            "Clean persistent_storage intermediates and retry."
        )
    run_cmd([sys.executable, converter, fp16_dir,
             "--outfile", f16, "--outtype", "f16"])
    _gguf_structural_check(f16)
    return f16


@retry(tries=MAX_HF_UPLOAD_RETRIES, delay=5.0, label="HF gguf upload")
def _publish_gguf_hf(path: str) -> None:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN required for upload")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    fname = os.path.basename(path)
    log(f"[HF] uploading gguf/{fname}")
    remote_path = f"gguf/{fname}"
    api.upload_file(path_or_fileobj=path, path_in_repo=remote_path,
                    repo_id=HF_REPO_ID, repo_type="model")
    # Keep the canonical F16 GGUF under fp16_model/gguf_output as well.
    if fname == "shadow-ronin-4b-f16.gguf":
        api.upload_file(path_or_fileobj=path,
                        path_in_repo=f"fp16_model/gguf_output/{fname}",
                        repo_id=HF_REPO_ID, repo_type="model")
    # Verify remote metadata when the Hub API exposes file sizes.
    try:
        info = api.repo_info(repo_id=HF_REPO_ID, repo_type="model", files_metadata=True)
        target = None
        for item in getattr(info, "siblings", []) or []:
            if getattr(item, "rfilename", None) == remote_path:
                target = item
                break
        remote_size = getattr(target, "size", None) if target else None
        local_size = os.path.getsize(path)
        if remote_size is not None and int(remote_size) != int(local_size):
            raise RuntimeError(
                f"Remote size mismatch for {remote_path}: local={local_size} remote={remote_size}"
            )
        if remote_size is not None:
            log(f"[HF] exact size verified: {remote_path} -> {remote_size} bytes")
        else:
            log(f"[HF] uploaded: {remote_path} (remote size metadata unavailable)")
    except RuntimeError:
        raise
    except Exception as exc:
        log(f"[HF] remote size verification warning: {exc}")


def _download_gguf_from_hf(fname: str, dest: str) -> str:
    """Download a single gguf/<fname> from HF into dest (file path)."""
    token = get_hf_token()
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    log(f"[HF] downloading gguf/{fname} -> {dest}")
    tmp = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"gguf/{fname}",
        repo_type="model",
        token=token,
        local_dir=os.path.dirname(dest),
        local_dir_use_symlinks=False,
    )
    if os.path.abspath(tmp) != os.path.abspath(dest):
        if os.path.isfile(dest):
            os.remove(dest)
        shutil.move(tmp, dest)
    _gguf_structural_check(dest)
    return dest


def _cleanup_quantized_artifact(path: str) -> None:
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
            log(f"[GGUF cleanup] removed local: {path}")
    except Exception as exc:
        raise RuntimeError(f"Could not clean {path}: {exc}") from exc
    _deep_cleanup(f"after cleanup {os.path.basename(path)}")


def _cleanup_fp16_dir(fp16_dir: str) -> None:
    """Remove recovered FP16 tree after F16 is safely on HF (frees ~8 GB)."""
    if not fp16_dir or not os.path.isdir(fp16_dir):
        return
    try:
        shutil.rmtree(fp16_dir, ignore_errors=True)
        log(f"[GGUF cleanup] removed local FP16 dir: {fp16_dir}")
        log(f"[disk] free after FP16 cleanup: {_free_disk_mb():.0f} MB")
    except Exception as exc:
        log(f"[GGUF cleanup] warn: could not remove FP16 dir: {exc}")


def export_and_quantize_gguf(fp16_dir: str) -> List[str]:
    """bnb-free GGUF pipeline — HF-first, clean after every stage.

    Disk policy:
      - Produce F16 → smoke → upload to HF → delete local FP16 (~8 GB)
      - For each quant: ensure F16 local (reuse or re-download from HF) →
        quantize → smoke → upload → delete that quant
      - After all quants: delete local F16
      - Never keep all quants on disk at once
    """
    repo_dir = os.path.join(PERMANENT_BASE_DIR, "llama.cpp")
    cli = ensure_llama_cpp()
    converter = find_llama_convert(repo_dir)
    quantizer = find_quantizer(repo_dir)
    _deep_cleanup("before GGUF pipeline")
    log(f"[disk] free at GGUF start: {_free_disk_mb():.0f} MB")

    state = _load_gguf_state()
    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    f16 = os.path.join(GGUF_OUTPUT_DIR, "shadow-ronin-4b-f16.gguf")
    f16_name = "shadow-ronin-4b-f16.gguf"

    # ---- F16 stage ----
    if state.get("stages", {}).get("F16", {}).get("status") == "PUBLISHED":
        log("[GGUF] F16 already PUBLISHED on HF")
        try:
            if os.path.isfile(f16):
                _gguf_structural_check(f16)
                log("[GGUF] reusing local F16 for remaining quants")
            else:
                f16 = None  # will fetch on demand from HF
        except Exception:
            f16 = None
    else:
        # Need to build F16 from FP16
        if not fp16_dir or not os.path.isdir(fp16_dir):
            raise RuntimeError("FP16 dir required to produce F16 GGUF")
        f16 = _ensure_f16_gguf(fp16_dir, converter)
        _mark_gguf_stage("F16", "VALIDATING", f16)
        smoke_test_gguf(cli, f16, "What is a compliance checklist?")
        log("=" * 72)
        log("SEQUENTIAL PUBLISH: F16")
        log("=" * 72)
        _publish_gguf_hf(f16)
        _mark_gguf_stage("F16", "PUBLISHED", f16)
        log("SEQUENTIAL PUBLISH COMPLETE: F16 -> HF")
        # FP16 no longer needed once F16 is on HF
        _cleanup_fp16_dir(fp16_dir)

    outputs: List[str] = []
    state = _load_gguf_state()

    # ---- quant stages ----
    for quant in QUANT_TARGETS:
        out = os.path.join(GGUF_OUTPUT_DIR, f"shadow-ronin-4b-{quant.lower()}.gguf")
        stage = state.get("stages", {}).get(quant, {})
        if stage.get("status") == "PUBLISHED":
            log(f"[GGUF] {quant}: already PUBLISHED on HF; skip")
            continue

        # Ensure F16 is local
        need_f16 = True
        if f16 and os.path.isfile(f16):
            try:
                _gguf_structural_check(f16)
                need_f16 = False
            except Exception:
                need_f16 = True
        if need_f16:
            f16 = os.path.join(GGUF_OUTPUT_DIR, f16_name)
            free = _free_disk_mb()
            log(f"[disk] free before F16 download: {free:.0f} MB")
            if 0 <= free < 9000:
                raise RuntimeError(
                    f"Not enough free disk ({free:.0f} MB) to download F16. "
                    "Clean local intermediates and retry."
                )
            _download_gguf_from_hf(f16_name, f16)
            log(f"[GGUF] F16 ready at {f16}")

        if os.path.isfile(out):
            try:
                os.remove(out)
            except Exception as exc:
                raise RuntimeError(f"Cannot remove stale {quant}: {exc}")

        _mark_gguf_stage(quant, "QUANTIZING", out)
        log(f"[GGUF] quantizing -> {quant}")
        log(f"[disk] free before quant {quant}: {_free_disk_mb():.0f} MB")
        try:
            run_cmd([quantizer, f16, out, quant])
            _gguf_structural_check(out)
            _mark_gguf_stage(quant, "VALIDATING", out)
            smoke_test_gguf(cli, out,
                            "Give one short security check explanation.")
            log("=" * 72)
            log(f"SEQUENTIAL PUBLISH: {quant}")
            log("=" * 72)
            _publish_gguf_hf(out)
            _mark_gguf_stage(quant, "PUBLISHED", out)
            outputs.append(out)
            _cleanup_quantized_artifact(out)
            log(f"[GGUF] {quant}: PUBLISHED + VERIFIED + CLEANED")
            log(f"[disk] free after {quant}: {_free_disk_mb():.0f} MB")
        except Exception as exc:
            _mark_gguf_stage(quant, "FAILED", out, f"{type(exc).__name__}: {exc}")
            raise
        state = _load_gguf_state()

    # Final: drop local F16 — it lives on HF now
    if f16 and os.path.isfile(f16):
        _cleanup_quantized_artifact(f16)
    _deep_cleanup("after GGUF pipeline")
    log(f"[disk] free at GGUF end: {_free_disk_mb():.0f} MB")
    return outputs


def verify_remote_release_exact() -> None:
    """Final HF verification: required paths + local/remote size checks where API exposes size."""
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF token required for final release verification")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    info = api.repo_info(repo_id=HF_REPO_ID, repo_type="model", files_metadata=True)
    files = {getattr(x, "rfilename", ""): x for x in (getattr(info, "siblings", []) or [])}

    required = [
        "PUBLICATION_MANIFEST.json",
        "gguf/shadow-ronin-4b-f16.gguf",
        "fp16_model/gguf_output/shadow-ronin-4b-f16.gguf",
    ]
    for q in QUANT_TARGETS:
        required.append(f"gguf/shadow-ronin-4b-{q.lower()}.gguf")
    missing = [p for p in required if p not in files]
    if missing:
        raise RuntimeError("Remote release missing: " + ", ".join(missing))

    log(f"[final verify] remote required artifacts present: {len(required)}")
    for p in required:
        size = getattr(files[p], "size", None)
        if size is not None:
            log(f"[final verify] {p}: {size} bytes")
    _set_master_stage("FINAL_RELEASE_VERIFY", "DONE", required_files=len(required))


def validate_sequential_release(fp16_dir: Optional[str] = None) -> None:
    """Validate that every stage is PUBLISHED on HF (local FP16 optional)."""
    state = _load_gguf_state()
    stages = state.get("stages", {})
    missing = [q for q in ["F16"] + list(QUANT_TARGETS)
               if stages.get(q, {}).get("status") != "PUBLISHED"]
    if missing:
        raise RuntimeError("Release incomplete; missing: " + ", ".join(missing))
    log("[release] Final sequential release validation: PASS")


# =============================================================================
# PHASE 12: RECOVERY MODE (bnb-free, trl-free, transformers-free)
# =============================================================================

def run_recovery_mode(recovery: Dict[str, Any], start_time: float) -> int:
    log("=" * 72)
    log("HF RECOVERY MODE")
    log("=" * 72)
    log(f"Repository: {HF_REPO_ID}")
    log(f"[disk] free at recovery start: {_free_disk_mb():.0f} MB")

    if recovery.get("gguf_published"):
        log(f"Found {len(recovery['gguf_published'])} GGUF file(s) on HF:")
        for f in recovery["gguf_published"]:
            log(f"  gguf/{f}")
        recover_gguf_stage_state(recovery["gguf_published"])
    else:
        log("No GGUF files on HF; will generate all quantizations.")

    if recovery.get("validation_available"):
        log(f"Recovering {len(recovery['validation_available'])} validation artifact(s)")
        recover_validation_artifacts(recovery["validation_available"])

    state = _load_gguf_state()
    f16_done = state.get("stages", {}).get("F16", {}).get("status") == "PUBLISHED"
    all_quants_done = all(
        state.get("stages", {}).get(q, {}).get("status") == "PUBLISHED"
        for q in QUANT_TARGETS
    )

    fp16_dir = None
    if f16_done and all_quants_done:
        log("All GGUF stages already PUBLISHED on HF — nothing to do.")
    elif f16_done:
        log("F16 already on HF; skipping FP16 download (will fetch F16 GGUF only if needed).")
        log("SFT: SKIPPED - recovered validated FP16 model (via F16 on HF)")
        log("GRPO: SKIPPED - recovered validated FP16 model")
        log("FP16 MERGE: SKIPPED - already available")
    else:
        # Need FP16 to produce F16
        log("Found fp16_model/ on HF; downloading only if local copy invalid...")
        log("Downloading/recovering FP16 artifacts...")
        fp16_dir = download_fp16_from_hf(DOWNLOADED_FP16_DIR)
        log(f"FP16 model at: {fp16_dir}")
        log("Validating FP16 model...")
        log("Validating tokenizer...")
        log("Validating chat template...")
        log("Validating safetensor shards...")
        validate_recovered_fp16_bnb_free(fp16_dir)
        log("HF FP16 RECOVERY: PASS")
        log("SFT: SKIPPED - recovered validated FP16 model")
        log("GRPO: SKIPPED - recovered validated FP16 model")
        log("FP16 MERGE: SKIPPED - already available")

    pip_install(["huggingface_hub>=0.30.0"])

    log("Resuming sequential GGUF release (HF-first, clean each stage)...")
    export_and_quantize_gguf(fp16_dir or DOWNLOADED_FP16_DIR)
    validate_sequential_release(fp16_dir)

    # Final local cleanup
    f16_src = os.path.join(GGUF_OUTPUT_DIR, "shadow-ronin-4b-f16.gguf")
    if os.path.isfile(f16_src):
        _cleanup_quantized_artifact(f16_src)
    if fp16_dir and os.path.isdir(fp16_dir):
        _cleanup_fp16_dir(fp16_dir)

    _write_runtime_compat_report()
    elapsed = time.time() - start_time
    log("=" * 72)
    log("PIPELINE COMPLETE (HF RECOVERY MODE)")
    log("=" * 72)
    log(f"[disk] free at end: {_free_disk_mb():.0f} MB")
    state = _load_gguf_state()
    for q in ["F16"] + list(QUANT_TARGETS):
        it = state.get("stages", {}).get(q, {})
        log(f"GGUF: {q} -> {it.get('status', 'UNKNOWN')}")
    log(f"Elapsed: {elapsed/60:.1f} minutes")
    log(f"HF REPO: https://huggingface.co/{HF_REPO_ID}")
    return 0


# =============================================================================
# PHASE 13: BITSANDBYTES SUBPROCESS PROBE + REPAIR
# =============================================================================

def _probe_bitsandbytes_subprocess() -> Tuple[bool, str]:
    """Import bitsandbytes in a FRESH subprocess. Returns (ok, output)."""
    probe = (
        "import sys\n"
        "try:\n"
        "    import bitsandbytes as bnb\n"
        "    print('BNB_VERSION=' + getattr(bnb, '__version__', 'unknown'))\n"
        "    print('BNB_OK')\n"
        "except Exception as e:\n"
        "    print('BNB_ERROR: ' + type(e).__name__ + ': ' + str(e))\n"
        "    import traceback\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", probe],
            text=True, capture_output=True, timeout=90,
        )
        out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        ok = r.returncode == 0 and "BNB_OK" in out
        return ok, out[-4000:]
    except Exception as exc:
        return False, f"probe exception: {exc}"


def _is_duplicate_registration_error(text: str) -> bool:
    t = text.lower()
    return (
        "duplicate registration" in t
        or "tried to register an operator" in t
        or ("already registered" in t and "operator" in t)
    )


def _is_abi_error(text: str) -> bool:
    t = text.lower()
    return (
        "undefined symbol" in t
        or ("abi" in t and "mismatch" in t)
    )


def _probe_stack_versions_subprocess() -> Dict[str, str]:
    probe = (
        "import json, importlib\n"
        "out = {}\n"
        "for name in ('torch','transformers','tokenizers','trl','peft',"
        "'huggingface_hub','bitsandbytes','accelerate','datasets',"
        "'sentence_transformers','psutil','numpy'):\n"
        "    try:\n"
        "        m = importlib.import_module(name)\n"
        "        out[name] = getattr(m, '__version__', 'unknown')\n"
        "    except Exception:\n"
        "        out[name] = None\n"
        "print('VERSIONS=' + json.dumps(out))\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", probe],
                           text=True, capture_output=True, timeout=90)
        for line in (r.stdout or "").splitlines():
            if line.startswith("VERSIONS="):
                return json.loads(line[len("VERSIONS="):])
    except Exception as exc:
        log(f"[probe] versions failed: {exc}")
    return {}


def _repair_bitsandbytes_for_stack() -> None:
    log("[bnb-repair] uninstalling broken bitsandbytes")
    _pip_uninstall(["bitsandbytes"])
    try:
        run_cmd([sys.executable, "-m", "pip", "cache", "purge", "bitsandbytes"],
                check=False)
    except Exception:
        pass
    spec = PIN_MATRIX["bitsandbytes"]
    log(f"[bnb-repair] installing {spec}")
    if not pip_install([spec]):
        log("[bnb-repair] falling back to newest bitsandbytes")
        if not pip_install(["bitsandbytes"]):
            raise RuntimeError("Could not reinstall bitsandbytes")


def handle_bitsandbytes_preflight() -> None:
    """
    Called BEFORE any trl/transformers import.
    Detects broken bnb and (repair + marker + _PipelineRestartRequired).
    """
    log("=" * 72)
    log("BITSANDBYTES PREFLIGHT (subprocess probe)")
    log("=" * 72)

    marker = _read_marker()
    if marker and marker.get("status") == BNB_REPAIRED_MARKER:
        log("[bnb-preflight] previous run repaired bnb; verifying fresh state")
        ok, out = _probe_bitsandbytes_subprocess()
        _compat_record("bnb_probe", "post_repair_ok", ok)
        _compat_record("bnb_probe", "post_repair_output", out[-1000:])
        if ok:
            log("[bnb-preflight] bitsandbytes now imports cleanly; clearing marker")
            _clear_marker()
            return
        _compat_error(f"bitsandbytes still broken after repair: {out[-500:]}")
        raise RuntimeError(
            "bitsandbytes still broken after repair + kernel restart. "
            f"Manual intervention required. Probe output: {out[-800:]}"
        )

    ok, out = _probe_bitsandbytes_subprocess()
    _compat_record("bnb_probe", "ok", ok)
    _compat_record("bnb_probe", "output", out[-2000:])
    log(f"[bnb-preflight] probe ok={ok}")

    if ok:
        log("[bnb-preflight] bitsandbytes is healthy")
        return

    if _is_duplicate_registration_error(out):
        _compat_warn("bitsandbytes duplicate operator registration detected")
        failure_mode = "duplicate_registration"
    elif _is_abi_error(out):
        _compat_warn("bitsandbytes ABI mismatch detected")
        failure_mode = "abi_mismatch"
    else:
        _compat_warn("bitsandbytes probe failed for unknown reason")
        failure_mode = "unknown"

    stack = _probe_stack_versions_subprocess()
    log(f"[bnb-preflight] stack snapshot: {stack}")

    log(f"[bnb-preflight] failure mode = {failure_mode}; repairing")
    try:
        _repair_bitsandbytes_for_stack()
    except Exception as exc:
        _compat_error(f"bitsandbytes repair failed: {exc}")
        raise RuntimeError(f"Cannot repair bitsandbytes: {exc}") from exc

    log("[bnb-preflight] verifying repair via subprocess")
    ok2, out2 = _probe_bitsandbytes_subprocess()
    _compat_record("bnb_probe", "post_repair_ok", ok2)

    _write_marker({
        "status": BNB_REPAIRED_MARKER,
        "failure_mode": failure_mode,
        "repair_output": out2[-1500:],
        "stack_before": stack,
        "reason": ("bitsandbytes torch operator registration cannot be repaired "
                   "in-process once the module has been loaded."),
    })
    _compat_record("restart_events", "bnb_repair",
                   {"failure_mode": failure_mode, "at": time.time()})

    log("=" * 72)
    log("KERNEL RESTART REQUIRED")
    log("=" * 72)
    log("bitsandbytes was reinstalled. Because a kernel restart is the only")
    log("reliable way to clear torch operator registration, the pipeline is")
    log("stopping cleanly now.")
    log("")
    log("ACTION REQUIRED:")
    log("  1. Restart the Kaggle kernel.")
    log("  2. Re-run this same script.")
    log("  3. The pipeline will read its marker and continue automatically.")
    raise _PipelineRestartRequired(
        "bitsandbytes was repaired; kernel restart required"
    )


# =============================================================================
# PHASE 14: KWARG FILTERING
# =============================================================================

def _filter_kwargs(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls_or_fn.__init__)
        params = sig.parameters
        for p in params.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return dict(kwargs)


def _tokenizer_kwarg(trainer_cls, tokenizer) -> Dict[str, Any]:
    try:
        sig = inspect.signature(trainer_cls.__init__)
        if "processing_class" in sig.parameters:
            return {"processing_class": tokenizer}
        if "tokenizer" in sig.parameters:
            return {"tokenizer": tokenizer}
    except Exception:
        pass
    return {}


def _build_sft_config(SFTConfig, base_kwargs: Dict[str, Any]):
    attempts = [
        ("max_length", dict(base_kwargs)),
        ("max_seq_length", {**base_kwargs,
                            "max_seq_length": base_kwargs.get("max_length", 512)}),
    ]
    last_exc = None
    for name, kwargs in attempts:
        try:
            if name == "max_seq_length":
                kwargs.pop("max_length", None)
            cfg = SFTConfig(**_filter_kwargs(SFTConfig, kwargs))
            _compat_record("sft_compat", "used_key", name)
            return cfg
        except TypeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Could not build SFTConfig: {last_exc}")


# =============================================================================
# PHASE 15: TRAINING-STACK DEPENDENCIES (bnb EXCLUDED)
# =============================================================================

def _probe_import(name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def _purge_python_modules(prefixes: tuple) -> None:
    """Purge ONLY the given prefixes. Never pass 'bitsandbytes' here."""
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _choose_pins_for_stack() -> Dict[str, str]:
    tf = _probe_stack_versions_subprocess().get("transformers") or ""
    if tf.startswith("5."):
        log(f"[pins] transformers {tf}; using 5.x pins")
        return {"transformers": PIN_MATRIX["transformers_5"],
                "tokenizers": PIN_MATRIX["tokenizers_5"],
                "trl": PIN_MATRIX["trl_5"], "hub": PIN_MATRIX["hub_1"]}
    if tf.startswith(("4.57", "4.58")):
        log(f"[pins] transformers {tf}; using 4.5x pins")
        return {"transformers": PIN_MATRIX["transformers_4"],
                "tokenizers": PIN_MATRIX["tokenizers_4"],
                "trl": PIN_MATRIX["trl_4"], "hub": PIN_MATRIX["hub_0"]}
    log(f"[pins] transformers '{tf}'; using 5.x default")
    return {"transformers": PIN_MATRIX["transformers_5"],
            "tokenizers": PIN_MATRIX["tokenizers_5"],
            "trl": PIN_MATRIX["trl_5"], "hub": PIN_MATRIX["hub_1"]}


def _install_training_stack() -> None:
    """Install training stack EXCLUDING bitsandbytes (handled separately)."""
    log("=" * 72)
    log("TRAINING-STACK INSTALLATION (bnb excluded)")
    log("=" * 72)
    pins = _choose_pins_for_stack()
    core = [
        pins["transformers"], pins["tokenizers"], pins["trl"], pins["hub"],
        PIN_MATRIX["peft"], PIN_MATRIX["datasets"], PIN_MATRIX["accelerate"],
        PIN_MATRIX["sentence_transformers"], PIN_MATRIX["sentencepiece"],
        PIN_MATRIX["gguf"], PIN_MATRIX["protobuf"], PIN_MATRIX["psutil"],
    ]
    log(f"Installing: {core}")
    if not pip_install(core):
        _compat_warn("bulk install failed; retrying individually")
        for spec in core:
            pip_install([spec])
    _purge_python_modules(("transformers", "tokenizers", "trl", "peft",
                           "sentence_transformers", "datasets"))


def _preflight_hf_apis_after_import() -> None:
    """Inspect critical APIs after training stack is installed."""
    try:
        import transformers
        _compat_record("transformers", "version", transformers.__version__)
        for sym in ("AutoTokenizer", "AutoModelForCausalLM",
                    "BitsAndBytesConfig", "TrainerCallback"):
            if not hasattr(transformers, sym):
                _compat_error(f"transformers missing {sym}")
    except Exception as exc:
        _compat_error(f"transformers import failed: {exc}")

    try:
        import tokenizers
        _compat_record("tokenizers", "version", tokenizers.__version__)
    except Exception as exc:
        _compat_error(f"tokenizers import failed: {exc}")

    try:
        import huggingface_hub
        _compat_record("huggingface_hub", "version",
                       getattr(huggingface_hub, "__version__", "unknown"))
    except Exception as exc:
        _compat_warn(f"huggingface_hub import issue: {exc}")

    try:
        import peft
        _compat_record("peft", "version", peft.__version__)
        for sym in ("LoraConfig", "PeftModel", "prepare_model_for_kbit_training"):
            if not hasattr(peft, sym):
                _compat_error(f"peft missing {sym}")
    except Exception as exc:
        _compat_error(f"peft import failed: {exc}")

    try:
        import trl
        _compat_record("trl", "version", trl.__version__)
        for sym in ("SFTConfig", "SFTTrainer", "GRPOConfig", "GRPOTrainer"):
            if not hasattr(trl, sym):
                _compat_error(f"trl missing {sym}")
        try:
            sft_params = inspect.signature(trl.SFTConfig.__init__).parameters
            for k in ("max_length", "max_seq_length", "packing", "loss_type",
                      "dataset_text_field", "warmup_ratio", "warmup_steps"):
                _compat_record("trl", f"sft_has_{k}", k in sft_params)
        except Exception:
            pass
        try:
            grpo_params = inspect.signature(trl.GRPOConfig.__init__).parameters
            for k in ("max_completion_length", "num_generations", "use_vllm",
                      "loss_type", "beta", "max_prompt_length",
                      "generation_batch_size"):
                _compat_record("trl", f"grpo_has_{k}", k in grpo_params)
        except Exception:
            pass
    except Exception as exc:
        _compat_error(f"trl import failed: {exc}")

    log(f"[runtime-compat] transformers="
        f"{_runtime_compat_report['transformers'].get('version')} "
        f"trl={_runtime_compat_report['trl'].get('version')} "
        f"peft={_runtime_compat_report['peft'].get('version')}")


def _check_torchvision_abi() -> Tuple[bool, str]:
    probe = "import torch, torchvision\nfrom torchvision.transforms import InterpolationMode\nprint('ok')\n"
    try:
        r = subprocess.run([sys.executable, "-c", probe],
                           text=True, capture_output=True, timeout=60)
        out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        return r.returncode == 0, out[-2000:]
    except Exception as exc:
        return False, repr(exc)


def _disable_torchvision_for_transformers() -> None:
    log("[torchvision] disabling module (ABI mismatch)")
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            sys.modules.pop(name, None)
    import importlib.abc

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torchvision" or fullname.startswith("torchvision."):
                raise ImportError("torchvision disabled (ABI mismatch).")
            return None

    for finder in list(sys.meta_path):
        if isinstance(finder, _Blocker):
            sys.meta_path.remove(finder)
    sys.meta_path.insert(0, _Blocker())
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"


def _neutralize_torchao() -> None:
    try:
        import peft.import_utils as peft_import_utils
        import importlib.metadata as md
        from packaging.version import Version
    except Exception:
        return

    def _safe_is_torchao_available() -> bool:
        try:
            installed = md.version("torchao")
        except Exception:
            return False
        minimum = getattr(peft_import_utils, "TORCHAO_MINIMUM_VERSION", "0.16.0")
        try:
            return Version(installed) >= Version(minimum)
        except Exception:
            return False

    peft_import_utils.is_torchao_available = _safe_is_torchao_available
    for modname in ("peft.tuners.lora.torchao",
                    "peft.tuners.lora.model",
                    "peft.tuners.lora.layer"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "is_torchao_available"):
                mod.is_torchao_available = _safe_is_torchao_available
        except Exception:
            pass
    log("[torchao] neutralized peft's strict check")


# =============================================================================
# PHASE 16: DATA EXTRACTION
# =============================================================================

QUESTION_KEYS = ["question", "query", "prompt", "instruction", "user", "input",
                 "request", "problem", "task"]
ANSWER_KEYS = ["answer", "response", "output", "assistant", "completion",
               "target", "solution"]
REASONING_KEYS = ["reasoning", "thinking", "cot", "chain_of_thought",
                  "rationale", "thoughts", "explanation"]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def safe_scalar(value: Any, max_chars: int = 6000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:max_chars]
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return canonical_json(value)[:max_chars]
    except Exception:
        return str(value)[:max_chars]


def iter_jsonl_records(path: str) -> Iterable[Tuple[int, Any, Optional[str]]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_number, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            try:
                yield line_number, json.loads(raw), None
            except Exception as exc:
                yield line_number, None, str(exc)


def sha256_file(path: str) -> str:
    return _sha256_file(path)


def first_nonempty(record: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in record:
            v = safe_scalar(record[key])
            if v:
                return v
    return ""


def extract_query_answer(record: Dict[str, Any]) -> Tuple[str, str]:
    messages = record.get("messages")
    if isinstance(messages, list):
        users, assistants = [], []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).lower()
            content = safe_scalar(item.get("content", ""))
            if role == "user" and content:
                users.append(content)
            elif role in ("assistant", "model") and content:
                assistants.append(content)
        if users or assistants:
            return (users[-1] if users else "",
                    assistants[-1] if assistants else "")
    return (first_nonempty(record, QUESTION_KEYS),
            first_nonempty(record, ANSWER_KEYS))


def _find_reasoning_field(record: Dict[str, Any]) -> str:
    return first_nonempty(record, REASONING_KEYS)


_REASONING_PATTERN = re.compile(
    rf"{re.escape(THINK_OPEN)}(.*?){re.escape(THINK_CLOSE)}\s*"
    rf"{re.escape(ANSWER_OPEN)}(.*?){re.escape(ANSWER_CLOSE)}",
    re.DOTALL,
)


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    m = _REASONING_PATTERN.search(text)
    if m:
        return m.group(2).strip()
    m = re.search(rf"{re.escape(ANSWER_OPEN)}(.*?){re.escape(ANSWER_CLOSE)}",
                  text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _wrap_answer_with_reasoning(answer: str, reasoning: str) -> str:
    answer = answer.strip()
    reasoning = reasoning.strip()
    if _REASONING_PATTERN.search(answer):
        return answer
    if reasoning:
        return (f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n"
                f"{ANSWER_OPEN}\n{answer}\n{ANSWER_CLOSE}")
    return f"{ANSWER_OPEN}\n{answer}\n{ANSWER_CLOSE}"


def render_chat(tokenizer, messages, add_generation_prompt=False) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt)


def build_prompt(tokenizer, query: str, context: Optional[str] = None) -> str:
    system = SYSTEM_PROMPT
    if context:
        system += ("\n\nUse the following retrieved knowledge when relevant. "
                   "Do not invent unsupported facts.\n\n" + context)
    return render_chat(tokenizer, [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ], add_generation_prompt=True)


def build_training_example(tokenizer, query: str, answer: str) -> str:
    return render_chat(tokenizer, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])


def standardize_to_chat(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    query, answer = extract_query_answer(example)
    if not query:
        query = first_nonempty(example, ["text", "content", "title", "description"])
    if not answer:
        answer = first_nonempty(example, ["text", "content", "description", "solution"])
    if not query:
        query = "Please analyze the supplied information."
    if not answer:
        answer = "I do not have enough information to provide a grounded answer."
    reasoning = _find_reasoning_field(example)
    answer = _wrap_answer_with_reasoning(answer, reasoning)
    return {"text": build_training_example(tokenizer, query, answer),
            "query": query, "answer": _extract_final_answer(answer)}


def locate_data_files(required: bool = True) -> List[str]:
    roots = [SPECIFIC_KAGGLE_DATASET_PATH, "/kaggle/input", DOWNLOADED_DATA_DIR]
    found, seen = [], set()
    for root in roots:
        if not os.path.exists(root):
            continue
        for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
            path = os.path.abspath(path)
            if path not in seen:
                seen.add(path)
                found.append(path)
    found.sort()
    if not found and required:
        raise FileNotFoundError("No .jsonl files found.")
    log(f"Discovered {len(found)} JSONL file(s).")
    for p in found:
        log(f"  DATA: {p}")
    return found


def normalize_jsonl_file(source_path: str, output_path: str) -> Dict[str, Any]:
    valid, invalid = 0, 0
    invalid_lines = []
    with open(output_path, "w", encoding="utf-8") as out:
        for line_no, obj, error in iter_jsonl_records(source_path):
            if error:
                invalid += 1
                invalid_lines.append({"line": line_no, "error": error})
                continue
            if isinstance(obj, dict):
                clean = obj
            elif isinstance(obj, list):
                clean = {"data": obj}
            else:
                clean = {"text": str(obj)}
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")
            valid += 1
    return {"source": os.path.abspath(source_path),
            "normalized": os.path.abspath(output_path),
            "valid_records": valid, "invalid_lines": invalid,
            "invalid_details": invalid_lines[:1000],
            "source_sha256": _sha256_file(source_path)}


def validate_and_normalize_jsonl_files(
    data_files: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    os.makedirs(NORMALIZED_JSONL_DIR, exist_ok=True)
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "model_id": MODEL_ID, "files": [],
              "total_valid_records": 0, "total_invalid_lines": 0}
    normalized_files = []
    for source in data_files:
        digest = _sha256_file(source)[:16]
        name = Path(source).stem
        out = os.path.join(NORMALIZED_JSONL_DIR,
                           f"{name}.{digest}.normalized.jsonl")
        result = normalize_jsonl_file(source, out)
        report["files"].append(result)
        report["total_valid_records"] += result["valid_records"]
        report["total_invalid_lines"] += result["invalid_lines"]
        if result["valid_records"] > 0:
            normalized_files.append(out)
        if result["invalid_lines"]:
            log(f"WARNING: {source}: {result['invalid_lines']} malformed line(s).")
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    log(f"JSONL validation: {report['total_valid_records']} valid, "
        f"{report['total_invalid_lines']} malformed.")
    if not normalized_files:
        raise RuntimeError("No valid JSONL records were found.")
    return normalized_files, report


def get_or_create_dataset(tokenizer, normalized_files: List[str]):
    from datasets import load_dataset
    cache_meta = os.path.join(PERMANENT_DATASET_DIR, "dataset_cache_meta.json")
    cache_path = os.path.join(PERMANENT_DATASET_DIR, "train_cache.jsonl")
    source_hashes = {os.path.abspath(p): _sha256_file(p) for p in normalized_files}
    expected = {"model_id": MODEL_ID, "source_hashes": source_hashes}
    if os.path.exists(cache_meta) and os.path.exists(cache_path):
        try:
            with open(cache_meta, "r", encoding="utf-8") as fh:
                old = json.load(fh)
            if old == expected:
                log("Using validated dataset cache.")
                return load_dataset("json", data_files=cache_path, split="train")
        except Exception as exc:
            log(f"Cache invalid; rebuilding: {exc}")
    rows = []
    for source in normalized_files:
        for _, record, error in iter_jsonl_records(source):
            if error or not isinstance(record, dict):
                continue
            rows.append(standardize_to_chat(record, tokenizer))
    if not rows:
        raise RuntimeError("No usable training examples were produced.")
    with open(cache_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(cache_meta, "w", encoding="utf-8") as fh:
        json.dump(expected, fh, indent=2)
    log(f"Prepared {len(rows)} training examples.")
    return load_dataset("json", data_files=cache_path, split="train")


# =============================================================================
# PHASE 17: RAG
# =============================================================================

def infer_category(source_file: str, record: Dict[str, Any]) -> str:
    text = (os.path.basename(source_file) + " " + safe_scalar(record)).lower()
    if "bigfix" in text or "relevance" in text or "fixlet" in text:
        return "BigFix"
    if "disa" in text or "stig" in text or "v-" in text:
        return "DISA"
    if "cis" in text:
        return "CIS"
    return Path(source_file).stem


def record_to_rag_text(source_file: str, record_number: int,
                       record: Dict[str, Any]) -> str:
    query, answer = extract_query_answer(record)
    reasoning = _find_reasoning_field(record)
    parts = [f"Source file: {os.path.basename(source_file)}",
             f"Record: {record_number}"]
    if query:
        parts.append(f"Question: {query}")
    if answer:
        parts.append(f"Answer: {answer}")
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")
    excluded = set(QUESTION_KEYS + ANSWER_KEYS + REASONING_KEYS + ["messages"])
    meta = []
    for k, v in record.items():
        if k in excluded:
            continue
        r = safe_scalar(v, max_chars=3000)
        if r:
            meta.append(f"{k}: {r}")
    if meta:
        parts.append("Metadata:")
        parts.extend(meta)
    if len(parts) <= 2:
        parts.append("Record JSON: " + canonical_json(record)[:10000])
    return "\n".join(parts).strip()


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class PersistentRAG:
    SCHEMA_VERSION = 2

    def __init__(self, data_files: List[str], db_path: str = RAG_DB_PATH):
        from sentence_transformers import SentenceTransformer
        import numpy as _np
        self._np = _np
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.data_files = [os.path.abspath(x) for x in data_files]
        self.db_path = db_path
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self.sync()
        self._load_dense_cache()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS sources ("
                    " source_id TEXT PRIMARY KEY, source_path TEXT NOT NULL,"
                    " file_hash TEXT NOT NULL, updated_at REAL NOT NULL,"
                    " record_count INTEGER NOT NULL)")
        cur.execute("CREATE TABLE IF NOT EXISTS documents ("
                    " doc_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,"
                    " source_file TEXT NOT NULL, record_number INTEGER NOT NULL,"
                    " chunk_number INTEGER NOT NULL, category TEXT,"
                    " content TEXT NOT NULL, content_hash TEXT NOT NULL,"
                    " metadata_json TEXT, updated_at REAL NOT NULL)")
        cur.execute("CREATE TABLE IF NOT EXISTS embeddings ("
                    " doc_id TEXT PRIMARY KEY, dim INTEGER NOT NULL,"
                    " vector BLOB NOT NULL)")
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                    "USING fts5(doc_id UNINDEXED, content)")
        cur.execute("CREATE TABLE IF NOT EXISTS meta ("
                    " key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION)))
        cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("embedding_model", EMBEDDING_MODEL))
        self.conn.commit()

    @staticmethod
    def _source_id(path: str) -> str:
        return hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _doc_id(source_id: str, record_number: int, chunk_number: int) -> str:
        return hashlib.sha256(
            f"{source_id}:{record_number}:{chunk_number}".encode("utf-8")
        ).hexdigest()[:40]

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _delete_source(self, source_id: str) -> None:
        rows = self.conn.execute("SELECT doc_id FROM documents WHERE source_id=?",
                                 (source_id,)).fetchall()
        for (doc_id,) in rows:
            self.conn.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM documents_fts WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE source_id=?", (source_id,))

    def _upsert_document(self, source_id, source_file, record_number,
                         chunk_number, category, content, metadata):
        content_hash = self._content_hash(content)
        doc_id = self._doc_id(source_id, record_number, chunk_number)
        old = self.conn.execute("SELECT content_hash FROM documents WHERE doc_id=?",
                                (doc_id,)).fetchone()
        if old and old[0] == content_hash:
            return None
        self.conn.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
        self.conn.execute("DELETE FROM documents_fts WHERE doc_id=?", (doc_id,))
        self.conn.execute(
            "INSERT OR REPLACE INTO documents("
            " doc_id, source_id, source_file, record_number, chunk_number,"
            " category, content, content_hash, metadata_json, updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (doc_id, source_id, source_file, record_number, chunk_number,
             category, content, content_hash,
             json.dumps(metadata, ensure_ascii=False), time.time()))
        self.conn.execute("INSERT INTO documents_fts(doc_id, content) VALUES(?, ?)",
                          (doc_id, content))
        return doc_id

    def sync(self) -> None:
        log("Synchronizing RAG directly from JSONL...")
        current_sources = set()
        for source_path in self.data_files:
            source_id = self._source_id(source_path)
            current_sources.add(source_id)
            source_hash = _sha256_file(source_path)
            old = self.conn.execute("SELECT file_hash FROM sources WHERE source_id=?",
                                    (source_id,)).fetchone()
            if old and old[0] == source_hash:
                continue
            log(f"RAG source changed/new: {os.path.basename(source_path)}")
            self._delete_source(source_id)
            record_count = 0
            for line_no, record, error in iter_jsonl_records(source_path):
                if error or not isinstance(record, dict):
                    continue
                record_count += 1
                category = infer_category(source_path, record)
                text = record_to_rag_text(source_path, line_no, record)
                chunks = chunk_text(text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
                metadata = {"source_file": os.path.basename(source_path),
                            "source_path": source_path,
                            "record_number": line_no, "category": category}
                for chunk_number, chunk in enumerate(chunks):
                    self._upsert_document(source_id, os.path.basename(source_path),
                                          line_no, chunk_number, category,
                                          chunk, metadata)
            self.conn.execute(
                "INSERT OR REPLACE INTO sources("
                " source_id, source_path, file_hash, updated_at, record_count)"
                " VALUES(?,?,?,?,?)",
                (source_id, source_path, source_hash, time.time(), record_count))
            self.conn.commit()
        rows = self.conn.execute("SELECT source_id FROM sources").fetchall()
        for (source_id,) in rows:
            if source_id not in current_sources:
                self._delete_source(source_id)
                self.conn.execute("DELETE FROM sources WHERE source_id=?",
                                  (source_id,))
        self.conn.commit()
        self._embed_missing()
        self._write_manifest()
        count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        log(f"RAG sync complete: {count} document chunk(s).")

    def _embed_missing(self) -> None:
        rows = self.conn.execute(
            "SELECT d.doc_id, d.content FROM documents d"
            " LEFT JOIN embeddings e ON e.doc_id=d.doc_id"
            " WHERE e.doc_id IS NULL").fetchall()
        if not rows:
            return
        log(f"Embedding {len(rows)} new/changed RAG chunk(s)...")
        bs = 64
        for start in range(0, len(rows), bs):
            batch = rows[start:start + bs]
            texts = [x[1] for x in batch]
            vectors = self.encoder.encode(texts, batch_size=bs,
                                          normalize_embeddings=True,
                                          show_progress_bar=False)
            vectors = self._np.asarray(vectors, dtype=self._np.float32)
            for (doc_id, _), vector in zip(batch, vectors):
                self.conn.execute(
                    "INSERT OR REPLACE INTO embeddings(doc_id, dim, vector)"
                    " VALUES(?,?,?)",
                    (doc_id, int(vector.shape[0]), vector.tobytes()))
            self.conn.commit()

    def _load_dense_cache(self) -> None:
        rows = self.conn.execute(
            "SELECT doc_id, dim, vector FROM embeddings ORDER BY doc_id").fetchall()
        ids, vectors = [], []
        for doc_id, dim, blob in rows:
            v = self._np.frombuffer(blob, dtype=self._np.float32)
            if v.size != dim:
                continue
            ids.append(doc_id)
            vectors.append(v)
        self.dense_ids = ids
        self.dense_matrix = (
            self._np.vstack(vectors).astype(self._np.float32, copy=False)
            if vectors else self._np.empty((0, 0), dtype=self._np.float32))

    def _fts_query(self, query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_./:@+-]+", query.lower())
        tokens = [t.replace('"', "") for t in tokens if len(t) >= 2]
        if not tokens:
            return ""
        return " OR ".join(tokens[:24])

    def lexical_search(self, query: str, limit: int):
        q = self._fts_query(query)
        if not q:
            return []
        try:
            rows = self.conn.execute(
                "SELECT doc_id, bm25(documents_fts) FROM documents_fts"
                " WHERE documents_fts MATCH ?"
                " ORDER BY bm25(documents_fts) LIMIT ?", (q, limit)).fetchall()
            return [(d, 1.0 / (1.0 + max(0.0, float(s)))) for d, s in rows]
        except sqlite3.Error:
            return []

    def dense_search(self, query: str, limit: int):
        if not self.dense_ids:
            return []
        v = self.encoder.encode([query], normalize_embeddings=True,
                                show_progress_bar=False)[0].astype(self._np.float32)
        scores = self.dense_matrix @ v
        order = self._np.argsort(-scores)[:limit]
        return [(self.dense_ids[int(i)], float(scores[int(i)])) for i in order]

    def search(self, query: str, top_k: int = RAG_TOP_K):
        k = max(top_k * 5, 10)
        lex = self.lexical_search(query, k)
        dense = self.dense_search(query, k)
        rrf, lex_raw, dense_raw = {}, {}, {}
        for rank, (doc_id, score) in enumerate(lex, 1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (60.0 + rank)
            lex_raw[doc_id] = score
        for rank, (doc_id, score) in enumerate(dense, 1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (60.0 + rank)
            dense_raw[doc_id] = score
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in ranked:
            row = self.conn.execute(
                "SELECT source_file, record_number, chunk_number, category,"
                " content, metadata_json FROM documents WHERE doc_id=?",
                (doc_id,)).fetchone()
            if not row:
                continue
            (sf, rn, cn, cat, content, mj) = row
            results.append({
                "doc_id": doc_id, "rrf_score": float(score),
                "lexical_score": lex_raw.get(doc_id, 0.0),
                "dense_score": dense_raw.get(doc_id, 0.0),
                "source_file": sf, "record_number": rn, "chunk_number": cn,
                "category": cat, "content": content,
                "metadata": json.loads(mj or "{}"),
            })
        return results

    def build_context(self, query: str, top_k: int = RAG_TOP_K):
        results = self.search(query, top_k=top_k)
        blocks = [
            f"[SOURCE {i}]\nFile: {it['source_file']}\nRecord: {it['record_number']}\n"
            f"Category: {it['category']}\n{it['content']}"
            for i, it in enumerate(results, 1)
        ]
        return "\n\n".join(blocks), results

    def _write_manifest(self) -> None:
        rows = self.conn.execute(
            "SELECT source_id, source_path, file_hash, record_count, updated_at"
            " FROM sources ORDER BY source_path").fetchall()
        dc = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        ec = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        manifest = {"schema_version": self.SCHEMA_VERSION,
                    "embedding_model": EMBEDDING_MODEL,
                    "chunk_size": RAG_CHUNK_SIZE, "chunk_overlap": RAG_CHUNK_OVERLAP,
                    "document_count": dc, "embedding_count": ec,
                    "sources": [{"source_id": r[0], "source_path": r[1],
                                 "file_hash": r[2], "record_count": r[3],
                                 "updated_at": r[4]} for r in rows],
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(RAG_MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

    def smoke_test(self, queries=None) -> None:
        count = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count == 0:
            log("RAG smoke test: EMPTY index.")
            return
        if queries is None:
            queries = ["CIS security configuration", "DISA STIG remediation",
                       "BigFix relevance compliance check"]
        log("Running RAG retrieval smoke tests...")
        for q in queries:
            results = self.search(q, top_k=3)
            log(f"RAG query='{q}' -> {len(results)} result(s)")

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# =============================================================================
# PHASE 18: GRPO DATASET
# =============================================================================

def build_grpo_dataset(tokenizer, normalized_files: List[str],
                       max_examples: int = GRPO_MAX_EXAMPLES):
    from datasets import Dataset
    rows = []
    for source in normalized_files:
        for _, record, error in iter_jsonl_records(source):
            if error or not isinstance(record, dict):
                continue
            query, answer = extract_query_answer(record)
            if not query or not answer:
                continue
            rows.append({
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": query}],
                "reference_answer": _extract_final_answer(answer),
            })
            if len(rows) >= max_examples:
                break
        if len(rows) >= max_examples:
            break
    return Dataset.from_list(rows)


# =============================================================================
# PHASE 19: MODEL LOADING + SFT + GRPO + MERGE
# =============================================================================

def load_quantized_base(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if torch.cuda.is_available():
        compute_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                         else torch.float16)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0},
            trust_remote_code=True)
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=None,
        trust_remote_code=True)


def patch_forward_for_trl(model) -> None:
    fwd = getattr(model, "forward", None)
    if isinstance(fwd, functools.partial):
        original = fwd

        def compat(self, *args, **kwargs):
            return original(*args, **kwargs)

        model.forward = types.MethodType(compat, model)
        log("Applied TRL forward compat wrapper.")
    try:
        log(f"TRL forward signature ready: {inspect.signature(model.forward)}")
    except Exception as exc:
        log(f"Forward signature inspection warning: {exc}")


def fresh_lora_config():
    from peft import LoraConfig
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])


def _preflight_model_forward(model, tokenizer, label: str = "preflight") -> None:
    import torch
    section = _runtime_compat_report.setdefault("model_forward_preflight", {})
    section.setdefault("attempts", [])
    attempt: Dict[str, Any] = {"label": label}
    try:
        device = next(model.parameters()).device
        attempt["device"] = str(device)
        probe_ids = tokenizer("hi", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model(probe_ids)
        logits = getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError("forward returned no logits")
        attempt["logits_shape"] = list(logits.shape)
        attempt["dtype"] = str(logits.dtype)
        attempt["ok"] = True
        section["attempts"].append(attempt)
        section["ok"] = True
        log(f"[runtime-compat] model forward OK ({label}) "
            f"shape={attempt['logits_shape']} dtype={attempt['dtype']}")
    except Exception as exc:
        attempt["ok"] = False
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        section["attempts"].append(attempt)
        _compat_error(f"model forward failed ({label}): {exc}")
        raise


# =============================================================================
# MASTER PIPELINE STAGE STATE / RESUME HELPERS
# =============================================================================

MASTER_STATE_PATH = os.path.join(STATE_BASE_DIR, "master_pipeline_state.json")
SFT_STAGE_DIR = os.path.join(PERMANENT_CHECKPOINT_DIR, "sft_train")
GRPO_STAGE_DIR = os.path.join(PERMANENT_GRPO_CHECKPOINT_DIR, "grpo_train")
SFT_HF_REMOTE = "checkpoints/sft_latest"
GRPO_HF_REMOTE = "checkpoints/grpo_latest"

def _load_master_state() -> Dict[str, Any]:
    if not os.path.isfile(MASTER_STATE_PATH):
        return {"version": 2, "stages": {}}
    try:
        with open(MASTER_STATE_PATH, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            d.setdefault("version", 2)
            d.setdefault("stages", {})
            return d
    except Exception as exc:
        log(f"[master-state] read warning: {exc}")
    return {"version": 2, "stages": {}}

def _save_master_state(state: Dict[str, Any]) -> None:
    tmp = MASTER_STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, MASTER_STATE_PATH)

def _set_master_stage(name: str, status: str, **extra: Any) -> None:
    state = _load_master_state()
    item = {"status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    item.update(extra)
    state["stages"][name] = item
    _save_master_state(state)
    log(f"[MASTER STAGE] {name} -> {status}")

def _stage_is_done(name: str) -> bool:
    return _load_master_state().get("stages", {}).get(name, {}).get("status") == "DONE"

def _adapter_is_valid(path: str) -> bool:
    return bool(
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "adapter_config.json"))
        and any(
            os.path.isfile(os.path.join(path, x))
            for x in ("adapter_model.safetensors", "adapter_model.bin")
        )
    )

def _latest_valid_checkpoint(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if not m:
            continue
        if os.path.isfile(os.path.join(p, "trainer_state.json")):
            candidates.append((int(m.group(1)), p))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None

def _publish_checkpoint_tree(local_dir: str, remote_root: str) -> None:
    token = get_hf_token()
    if not token or not os.path.isdir(local_dir):
        log("[checkpoint upload] skipped: HF token or local directory unavailable")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    for root, _, files in os.walk(local_dir):
        for fname in sorted(files):
            if fname.endswith((".safetensors", ".bin", ".json", ".txt", ".pt", ".pth")):
                local = os.path.join(root, fname)
                rel = os.path.relpath(local, local_dir).replace(os.sep, "/")
                remote = f"{remote_root}/{rel}"
                try:
                    _hf_upload_file(api, local, remote)
                except Exception as exc:
                    log(f"[checkpoint upload] warning for {remote}: {exc}")

def _find_resume_checkpoint(output_dir: str) -> Optional[str]:
    latest = _latest_valid_checkpoint(output_dir)
    if latest:
        log(f"[resume] latest valid checkpoint: {latest}")
    return latest

def _safe_json_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception:
        return False


def train_sft(model_id: str, tokenizer, dataset) -> str:
    import torch
    from transformers import TrainerCallback
    from peft import prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    if _adapter_is_valid(SFT_STAGE_DIR):
        log(f"[SFT] final adapter already valid: {SFT_STAGE_DIR}")
        _publish_checkpoint_tree(SFT_STAGE_DIR, SFT_HF_REMOTE)
        _set_master_stage("SFT", "DONE", output_dir=SFT_STAGE_DIR, recovered=True)
        return SFT_STAGE_DIR

    os.makedirs(SFT_STAGE_DIR, exist_ok=True)
    resume_checkpoint = _find_resume_checkpoint(SFT_STAGE_DIR)

    class _CB(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            log(f"SFT TRAINING STARTED: step={state.global_step} total={state.max_steps}")
            return control
        def on_step_end(self, args, state, control, **kw):
            if state.global_step:
                log(f"SFT progress: {state.global_step}/{state.max_steps}")
            return control
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and logs.get("loss") is not None:
                log(f"SFT metrics: step={state.global_step} loss={float(logs['loss']):.5f}")
            return control

    model = trainer = None
    try:
        _set_master_stage("SFT", "RUNNING", output_dir=SFT_STAGE_DIR,
                          resume_checkpoint=resume_checkpoint)
        model = load_quantized_base(model_id)
        model.config.use_cache = False
        _preflight_model_forward(model, tokenizer, "sft_master")
        model = prepare_model_for_kbit_training(model)
        patch_forward_for_trl(model)

        sft_kwargs = dict(
            output_dir=SFT_STAGE_DIR, overwrite_output_dir=False,
            num_train_epochs=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, learning_rate=2e-4,
            logging_strategy="steps", logging_steps=1, logging_first_step=True,
            save_strategy="steps", save_steps=50, save_total_limit=3,
            report_to="none", disable_tqdm=False,
            fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            tf32=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit", max_grad_norm=1.0, warmup_ratio=0.03,
            weight_decay=0.0, max_length=SFT_MAX_LENGTH, packing=False,
            dataset_text_field="text", loss_type="nll",
            dataloader_num_workers=0, group_by_length=False,
            dataset_num_proc=1, max_steps=SFT_MAX_STEPS, seed=42)
        sft_config = _build_sft_config(SFTConfig, sft_kwargs)

        trainer_kwargs = dict(model=model, args=sft_config,
                              train_dataset=dataset, peft_config=fresh_lora_config(),
                              callbacks=[_CB()])
        trainer_kwargs.update(_tokenizer_kwarg(SFTTrainer, tokenizer))
        trainer = SFTTrainer(**_filter_kwargs(SFTTrainer, trainer_kwargs))

        log("SFT: entering trainer.train()")
        if resume_checkpoint and os.path.isdir(resume_checkpoint):
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()

        trainer.save_model(SFT_STAGE_DIR)
        tokenizer.save_pretrained(SFT_STAGE_DIR)

        if not _adapter_is_valid(SFT_STAGE_DIR):
            raise RuntimeError("SFT completed but final adapter is invalid")

        _publish_checkpoint_tree(SFT_STAGE_DIR, SFT_HF_REMOTE)
        _set_master_stage("SFT", "DONE", output_dir=SFT_STAGE_DIR,
                          resumed_from=resume_checkpoint)
        log(f"SFT COMPLETE: {SFT_STAGE_DIR}")
        return SFT_STAGE_DIR
    except Exception as exc:
        _set_master_stage("SFT", "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        _deep_cleanup("after SFT master stage")

def find_adapter_dir(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    if os.path.isdir(path):
        cands = []
        for name in os.listdir(path):
            child = os.path.join(path, name)
            if os.path.isdir(child) and os.path.isfile(
                os.path.join(child, "adapter_config.json")):
                cands.append(child)
        if cands:
            cands.sort(key=os.path.getmtime, reverse=True)
            return cands[0]
    raise FileNotFoundError(f"No adapter_config.json under: {path}")


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return safe_scalar(completion)
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                c = item.get("content")
                if c:
                    parts.append(str(c))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(completion)


def _train_grpo_once(model_id, tokenizer, grpo_dataset, sft_dir):
    from trl import GRPOConfig, GRPOTrainer
    from peft import PeftModel, prepare_model_for_kbit_training
    import torch
    from transformers import TrainerCallback

    if len(grpo_dataset) == 0:
        raise RuntimeError("GRPO dataset is empty.")

    output_dir = GRPO_STAGE_DIR
    os.makedirs(output_dir, exist_ok=True)
    resume_checkpoint = _find_resume_checkpoint(output_dir)

    _neutralize_torchao()
    model = trainer = None
    try:
        if len(grpo_dataset) > GRPO_MAX_EXAMPLES:
            grpo_dataset = grpo_dataset.select(range(GRPO_MAX_EXAMPLES))
        log(f"GRPO dataset: {len(grpo_dataset)} prompts")

        model = load_quantized_base(model_id)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        sft_adapter = find_adapter_dir(sft_dir)
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
        model.config.use_cache = False
        patch_forward_for_trl(model)

        def correctness_reward_func(prompts=None, completions=None, reference_answer=None, **kw):
            refs = reference_answer or kw.get("reference_answer") or []
            if not isinstance(refs, list):
                refs = [refs]
            scores = []
            for i, c in enumerate(completions or []):
                gen = _extract_final_answer(_completion_to_text(c)).lower().strip()
                ref = str(refs[i % len(refs)]).lower().strip() if refs else ""
                if not gen or not ref:
                    scores.append(0.0)
                    continue
                gt = set(re.findall(r"\w+", gen))
                rt = set(re.findall(r"\w+", ref))
                if not rt:
                    scores.append(0.0)
                    continue
                p = len(gt & rt) / max(1, len(gt))
                r = len(gt & rt) / max(1, len(rt))
                scores.append(float(min(1.0, 2*p*r/(p+r))) if p+r else 0.0)
            return scores

        def format_reward_func(completions=None, **kw):
            return [
                0.25 if THINK_OPEN in _completion_to_text(c)
                and THINK_CLOSE in _completion_to_text(c)
                and ANSWER_OPEN in _completion_to_text(c)
                and ANSWER_CLOSE in _completion_to_text(c)
                else 0.0
                for c in (completions or [])
            ]

        grpo_kwargs = dict(
            output_dir=output_dir, overwrite_output_dir=False,
            num_train_epochs=1, max_steps=GRPO_MAX_STEPS,
            per_device_train_batch_size=GRPO_BATCH_SIZE,
            gradient_accumulation_steps=GRPO_GRAD_ACCUM,
            learning_rate=5e-6, warmup_ratio=0.05,
            logging_strategy="steps", logging_steps=1,
            save_strategy="steps", save_steps=10, save_total_limit=3,
            report_to="none", remove_unused_columns=False,
            gradient_checkpointing=True,
            fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            max_prompt_length=GRPO_MAX_PROMPT_LENGTH,
            max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
            num_generations=GRPO_NUM_GENERATIONS,
            generation_batch_size=GRPO_NUM_GENERATIONS,
            beta=0.0, loss_type="bnpo", use_liger_loss=False,
            dataloader_num_workers=0, dataloader_pin_memory=False,
            temperature=0.8, top_p=0.95, repetition_penalty=1.05,
            use_vllm=False, log_completions=True,
            num_completions_to_print=1, seed=42)
        grpo_args = GRPOConfig(**_filter_kwargs(GRPOConfig, grpo_kwargs))

        class _CB(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                log(f"GRPO TRAINING STARTED: step={state.global_step} total={state.max_steps}")
                return control
            def on_step_end(self, args, state, control, **kw):
                log(f"GRPO progress: {state.global_step}/{state.max_steps}")
                return control
            def on_log(self, args, state, control, logs=None, **kw):
                if logs:
                    log(f"GRPO metrics: step={state.global_step} loss={logs.get('loss')} reward={logs.get('reward')}")
                return control

        trainer_kwargs = dict(model=model, args=grpo_args,
                              train_dataset=grpo_dataset,
                              reward_funcs=[correctness_reward_func, format_reward_func],
                              callbacks=[_CB()])
        trainer_kwargs.update(_tokenizer_kwarg(GRPOTrainer, tokenizer))
        trainer = GRPOTrainer(**_filter_kwargs(GRPOTrainer, trainer_kwargs))

        log("GRPO: entering trainer.train()")
        if resume_checkpoint and os.path.isdir(resume_checkpoint):
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        if not _adapter_is_valid(output_dir):
            raise RuntimeError("GRPO completed but final adapter is invalid")

        _publish_checkpoint_tree(output_dir, GRPO_HF_REMOTE)
        log(f"GRPO adapter saved: {output_dir}")
        return output_dir
    finally:
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        _deep_cleanup("after GRPO master stage")


def train_grpo(model_id, tokenizer, grpo_dataset, sft_dir):
    if _adapter_is_valid(GRPO_STAGE_DIR):
        log(f"[GRPO] final adapter already valid: {GRPO_STAGE_DIR}")
        _publish_checkpoint_tree(GRPO_STAGE_DIR, GRPO_HF_REMOTE)
        _set_master_stage("GRPO", "DONE", output_dir=GRPO_STAGE_DIR, recovered=True)
        return GRPO_STAGE_DIR
    try:
        _set_master_stage("GRPO", "RUNNING", output_dir=GRPO_STAGE_DIR)
        result = _train_grpo_once(model_id, tokenizer, grpo_dataset, sft_dir)
        _set_master_stage("GRPO", "DONE", output_dir=result)
        return result
    except Exception as exc:
        _set_master_stage("GRPO", "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise

def merge_to_fp16(base_model_id: str, adapter_dir: str, output_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    _neutralize_torchao()

    if _stage_is_done("FP16_MERGE") and os.path.isdir(output_dir):
        log(f"[merge] existing completed FP16 reused: {output_dir}")
        return output_dir

    if not _adapter_is_valid(adapter_dir):
        raise RuntimeError(f"Cannot merge: invalid adapter {adapter_dir}")

    _set_master_stage("FP16_MERGE", "RUNNING", output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Never delete an existing merge directory on restart. A partial directory
    # is repaired file-by-file by save_pretrained below.
    log("Loading base model in FP16 for LoRA merge.")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tok.save_pretrained(output_dir)

    validate_recovered_fp16_bnb_free(output_dir)
    _set_master_stage("FP16_MERGE", "DONE", output_dir=output_dir)
    del model, base, merged
    _deep_cleanup("after FP16 merge")
    log(f"FP16 merged model saved: {output_dir}")
    return output_dir

def smoke_test_hf_model(model_dir: str,
                        query: str = "Explain a secure configuration check.") -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log("Running FP16/HF smoke test BEFORE GGUF conversion.")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        raise RuntimeError("Merged tokenizer has no chat template.")
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, device_map=None,
            trust_remote_code=True)
    prompt = build_prompt(tok, query)
    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                     skip_special_tokens=True).strip()
    if not gen:
        raise RuntimeError("FP16/HF smoke test empty.")
    log("FP16/HF smoke test: PASS")
    log(f"FP16 sample output: {gen[:500]}")
    del model
    _deep_cleanup("after FP16/HF smoke test")
    return gen


# =============================================================================
# PHASE 20: PUBLISH
# =============================================================================

def _write_publication_manifest(path: str, hf_published: bool = False,
                                gguf_files: Optional[List[str]] = None) -> None:
    payload = {"model": MODEL_ID, "hf_repo": HF_REPO_ID,
               "quantizations": QUANT_TARGETS, "fp16_model": True,
               "gguf_files": [os.path.basename(p) for p in (gguf_files or [])
                              if os.path.isfile(p)],
               "validation_passed": True, "hf_published": hf_published,
               "published_at_epoch": time.time()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


@retry(tries=MAX_HF_UPLOAD_RETRIES, delay=5.0, label="HF upload_file")
def _hf_upload_file(api, local: str, remote: str) -> None:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=HF_REPO_ID, repo_type="model")


def publish_to_huggingface(fp16_dir, gguf_files, rag_db_path,
                           rag_manifest_path, validation_report_path) -> None:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN required.")
    from huggingface_hub import HfApi
    log("=" * 72)
    log("FINAL PUBLISH [HF]")
    log("=" * 72)
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)

    log("[HF] Uploading FP16 model files...")
    for fname in sorted(os.listdir(fp16_dir)):
        p = os.path.join(fp16_dir, fname)
        if os.path.isfile(p):
            _hf_upload_file(api, p, f"fp16_model/{fname}")

    log("[HF] Uploading GGUF files...")
    for path in gguf_files:
        if os.path.isfile(path):
            _hf_upload_file(api, path, f"gguf/{os.path.basename(path)}")

    log("[HF] Uploading validation artifacts...")
    with tempfile.TemporaryDirectory() as tmp:
        m = os.path.join(tmp, "PUBLICATION_MANIFEST.json")
        _write_publication_manifest(m, True, gguf_files)
        _hf_upload_file(api, m, "PUBLICATION_MANIFEST.json")
        r = os.path.join(tmp, "README.md")
        with open(r, "w", encoding="utf-8") as fh:
            fh.write(
                f"---\nlibrary_name: transformers\nbase_model: {MODEL_ID}\n---\n\n"
                f"# Shadow-Ronin-Rag-gguf\n\n"
                f"Validated Shadow-Ronin 4B release.\n\n"
                f"## Contents\n\n"
                f"- `fp16_model/` - merged validated FP16 model\n"
                f"- `gguf/` - validated GGUF quantizations\n"
                f"- `validation/` - validation + RAG artifacts\n"
                f"- `PUBLICATION_MANIFEST.json`\n\n"
                f"## Quantizations\n\n{', '.join(QUANT_TARGETS)}\n")
        _hf_upload_file(api, r, "README.md")

    for local, remote in [
        (validation_report_path, "jsonl_validation_report.json"),
        (rag_manifest_path, "manifest.json"),
        (rag_db_path, "rag.sqlite"),
        (RUNTIME_COMPAT_REPORT_PATH, "runtime_compatibility_report.json"),
        (HF_RECOVERY_REPORT_PATH, "hf_recovery_report.json"),
    ]:
        if os.path.isfile(local):
            _hf_upload_file(api, local, f"validation/{remote}")

    log(f"Hugging Face publish COMPLETE: https://huggingface.co/{HF_REPO_ID}")


# =============================================================================
# PHASE 21: SCRATCH MODE
# =============================================================================

# =============================================================================
# PHASE 21B: STAGED FP16 / CHECKPOINT / RUNTIME PUBLICATION
# =============================================================================

def _publish_fp16_only(fp16_dir: str) -> None:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF token required for staged FP16 publication")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    for root, _, files in os.walk(fp16_dir):
        for fname in sorted(files):
            local = os.path.join(root, fname)
            rel = os.path.relpath(local, fp16_dir).replace(os.sep, "/")
            _hf_upload_file(api, local, f"fp16_model/{rel}")
    _set_master_stage("FP16_PUBLISHED", "DONE", remote=HF_REPO_ID)

def _capture_runtime_bundle() -> Optional[str]:
    bundle = os.path.join(STATE_BASE_DIR, "runtime_bundle")
    envdir = os.path.join(bundle, "environment")
    mandir = os.path.join(bundle, "manifests")
    os.makedirs(envdir, exist_ok=True)
    os.makedirs(mandir, exist_ok=True)

    def write(name: str, value: str):
        Path(os.path.join(envdir, name)).write_text(value, encoding="utf-8")

    try:
        write("python_version.txt", sys.version)
        write("pip_freeze.txt", subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], text=True,
            capture_output=True, check=False).stdout)
        write("nvidia_smi.txt", subprocess.run(
            ["nvidia-smi"], text=True, capture_output=True,
            check=False).stdout)
        cuda = _detect_cuda_toolkit()
        write("cuda_info.json", json.dumps(cuda, indent=2))
        repo = os.path.join(PERMANENT_BASE_DIR, "llama.cpp")
        if os.path.isdir(repo):
            try:
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                    capture_output=True, check=False).stdout.strip()
            except Exception:
                commit = ""
        else:
            commit = ""
        # Copy the proven llama.cpp runtime executables, not the large build tree.
        runtime_bin = os.path.join(bundle, "llama.cpp")
        os.makedirs(runtime_bin, exist_ok=True)
        repo = os.path.join(PERMANENT_BASE_DIR, "llama.cpp")
        for candidate in (
            os.path.join(repo, "build_cuda", "bin", "llama"),
            os.path.join(repo, "build_cuda", "bin", "llama-cli"),
            os.path.join(repo, "build_cuda", "bin", "llama-quantize"),
            os.path.join(repo, "build_cuda", "bin", "quantize"),
            os.path.join(repo, "build_cpu", "bin", "llama"),
            os.path.join(repo, "build_cpu", "bin", "llama-cli"),
            os.path.join(repo, "build_cpu", "bin", "llama-quantize"),
        ):
            if os.path.isfile(candidate):
                shutil.copy2(candidate, os.path.join(runtime_bin, os.path.basename(candidate)))

        runtime = {
            "base_model": MODEL_ID,
            "model_repository": HF_REPO_ID,
            "runtime_repository": RUNTIME_HF_REPO_ID,
            "llama_cpp_commit": commit,
            "security": {"hf_token_saved": False, "credentials_saved": False},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        Path(os.path.join(mandir, "runtime_manifest.json")).write_text(
            json.dumps(runtime, indent=2), encoding="utf-8")
        Path(os.path.join(bundle, "README.md")).write_text(
            "# Shadow-Ronin Runtime Bundle\\n\\n"
            "Reproducibility bundle generated by the end-to-end pipeline. "
            "Secrets and credentials are intentionally excluded. "
            "This is not a byte-for-byte Kaggle VM snapshot.\\n",
            encoding="utf-8")
        return bundle
    except Exception as exc:
        log(f"[runtime bundle] capture warning: {exc}")
        return None

def _publish_runtime_bundle(bundle_dir: Optional[str]) -> None:
    if not bundle_dir or not os.path.isdir(bundle_dir):
        return
    token = get_hf_token()
    if not token:
        log("[runtime bundle] no HF token; skipping runtime repository publish")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=RUNTIME_HF_REPO_ID, repo_type="model", exist_ok=True)
    for root, _, files in os.walk(bundle_dir):
        for fname in sorted(files):
            local = os.path.join(root, fname)
            rel = os.path.relpath(local, bundle_dir).replace(os.sep, "/")
            try:
                api.upload_file(path_or_fileobj=local, path_in_repo=rel,
                                repo_id=RUNTIME_HF_REPO_ID, repo_type="model")
            except Exception as exc:
                log(f"[runtime bundle] upload warning {rel}: {exc}")
    log(f"[runtime bundle] published: https://huggingface.co/{RUNTIME_HF_REPO_ID}")


def run_scratch_mode(start_time: float, bnb_preflight_done: bool = False) -> int:
    log("=" * 72)
    log("SHADOW-RONIN MASTER DEVELOPMENT PIPELINE")
    log("SCRATCH + LOCAL RESUME + HF STAGE RECOVERY")
    log("=" * 72)

    if not bnb_preflight_done:
        handle_bitsandbytes_preflight()

    _install_training_stack()

    tv_ok, tv_detail = _check_torchvision_abi()
    if not tv_ok:
        _disable_torchvision_for_transformers()
        _compat_record("torchvision", "abi_ok", False)
        _compat_record("torchvision", "action", "disabled")
    else:
        _compat_record("torchvision", "abi_ok", True)
    _neutralize_torchao()

    _preflight_hf_apis_after_import()
    _write_runtime_compat_report()
    if RUNTIME_COMPAT_FAIL_FAST and _runtime_compat_report["errors"]:
        raise RuntimeError("Runtime preflight failed:\n  - " +
                           "\n  - ".join(_runtime_compat_report["errors"]))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if not getattr(tok, "chat_template", None):
        raise RuntimeError(f"{MODEL_ID} has no chat template")

    _set_master_stage("DATA", "RUNNING")
    raw = locate_data_files(required=True)
    normalized, report = validate_and_normalize_jsonl_files(raw)
    if report["total_valid_records"] == 0:
        raise RuntimeError("No valid JSONL records.")
    _set_master_stage("DATA", "DONE", valid_records=report["total_valid_records"])

    rag = None
    if not DISABLE_RAG:
        try:
            rag = PersistentRAG(data_files=normalized, db_path=RAG_DB_PATH)
            rag.smoke_test()
            _set_master_stage("RAG", "DONE", db_path=RAG_DB_PATH)
        except Exception as exc:
            _compat_warn(f"RAG setup failed (non-fatal): {exc}")
            _set_master_stage("RAG", "FAILED", error=str(exc))

    dataset = get_or_create_dataset(tok, normalized)
    sft_dir = train_sft(MODEL_ID, tok, dataset)

    grpo_dataset = build_grpo_dataset(tok, normalized)
    grpo_dir = train_grpo(MODEL_ID, tok, grpo_dataset, sft_dir)

    adapter = find_adapter_dir(grpo_dir)
    fp16_dir = merge_to_fp16(MODEL_ID, adapter, PERMANENT_EXPORT_DIR)
    smoke_test_hf_model(fp16_dir)

    if not _stage_is_done("FP16_PUBLISHED"):
        _publish_fp16_only(fp16_dir)
    else:
        log("[FP16] already published; skipping upload")

    # Preserve validation artifacts before the GGUF phase.
    publish_to_huggingface(
        fp16_dir=fp16_dir, gguf_files=[],
        rag_db_path=RAG_DB_PATH, rag_manifest_path=RAG_MANIFEST_PATH,
        validation_report_path=VALIDATION_REPORT_PATH)

    gguf_files = export_and_quantize_gguf(fp16_dir)
    validate_sequential_release(fp16_dir)
    verify_remote_release_exact()

    bundle = _capture_runtime_bundle()
    _publish_runtime_bundle(bundle)

    if rag is not None:
        try:
            rag.close()
        except Exception:
            pass

    _write_runtime_compat_report()
    elapsed = time.time() - start_time
    log("=" * 72)
    log("PIPELINE COMPLETE — SCRATCH/RESUME MODE")
    log("=" * 72)
    log(f"Elapsed: {elapsed/60:.1f} minutes")
    log(f"MODEL REPO: https://huggingface.co/{HF_REPO_ID}")
    log(f"RUNTIME REPO: https://huggingface.co/{RUNTIME_HF_REPO_ID}")
    return 0


# =============================================================================
# PHASE 22: ENTRY POINT (recovery-first)
# =============================================================================

def main() -> int:
    start = time.time()
    log("=" * 72)
    log("Shadow-Ronin 4B — Recovery-First, Restart-Aware Pipeline")
    log("=" * 72)
    log(f"MODEL_ID={MODEL_ID}")
    log(f"HF_REPO_ID={HF_REPO_ID}")
    log(f"FORCE_SCRATCH_TRAINING={FORCE_SCRATCH_TRAINING}")
    log(f"SKIP_HF_RECOVERY={SKIP_RECOVERY}")
    log(f"DISABLE_RAG={DISABLE_RAG}")
    log(f"RUNTIME_HF_REPO_ID={RUNTIME_HF_REPO_ID}")
    log("RECOVERY POLICY: HF -> local state -> checkpoints -> scratch")
    log("TRAINING POLICY: never retrain a valid completed stage")
    log("GGUF POLICY: every quant is generated directly from F16")
    log(f"SCRATCH_BASE_DIR={SCRATCH_BASE_DIR}  (large files; may be /kaggle/tmp)")
    log(f"STATE_BASE_DIR={STATE_BASE_DIR}  (JSON state only; persists)")
    log(f"[disk] scratch free≈{_free_disk_mb(SCRATCH_BASE_DIR):.0f} MB  "
        f"working free≈{_free_disk_mb(KAGGLE_WORKING_DIR):.0f} MB")
    _log_ram("[start] ")

    # --- Phase A: HF auth (huggingface_hub only) ---
    log("=" * 72)
    log("HF AUTHENTICATION")
    log("=" * 72)
    _ensure_huggingface_hub()
    authenticate_hf()

    # --- Phase C: HF RECOVERY CHECK — before ANY heavy imports ---
    if not FORCE_SCRATCH_TRAINING and not SKIP_RECOVERY:
        log("=" * 72)
        log("HF RECOVERY CHECK")
        log("=" * 72)
        try:
            recovery = check_hf_recovery()
        except Exception as exc:
            log(f"[HF recovery] probe failed: {exc}")
            recovery = {"available": False, "fp16_available": False,
                        "gguf_published": [], "validation_available": [],
                        "files": [], "probe_error": str(exc)}

        if recovery["fp16_available"]:
            return run_recovery_mode(recovery, start)

        if recovery["available"]:
            log("[HF recovery] repo reachable but no valid fp16_model/")
            log("[HF recovery] falling back to scratch training")
        else:
            log(f"[HF recovery] repo not usable ({recovery.get('probe_error')})")
            log("[HF recovery] falling back to scratch training")
    else:
        if FORCE_SCRATCH_TRAINING:
            log("[mode] FORCE_SCRATCH_TRAINING=1; skipping recovery")
        if SKIP_RECOVERY:
            log("[mode] SKIP_HF_RECOVERY=1; skipping recovery")

    # --- Phase D: scratch training ---
    # IMPORTANT: do the bnb subprocess preflight before importing torch in this
    # process. This prevents the exact duplicate torch-operator registration
    # failure seen in 22.log.
    log("=" * 72)
    log("PRE-TRAINING RUNTIME PRECHECK")
    log("=" * 72)
    handle_bitsandbytes_preflight()

    log("=" * 72)
    log("HARDWARE DETECTION")
    log("=" * 72)
    _detect_hardware()

    return run_scratch_mode(start, bnb_preflight_done=True)


# =============================================================================
# TOP-LEVEL LAUNCHER — clean exit, no IPython traceback spam
# =============================================================================

def _safe_main() -> int:
    try:
        return main()
    except _PipelineRestartRequired as exc:
        log("=" * 72)
        log("PIPELINE STOPPED: KERNEL RESTART REQUIRED")
        log("=" * 72)
        log(str(exc))
        log("")
        log("ACTION: Restart the Kaggle kernel and re-run this script.")
        log("        The pipeline will pick up where it left off.")
        return RESTART_EXIT_CODE
    except KeyboardInterrupt:
        log("[stop] interrupted by user")
        return 130
    except Exception as exc:
        log("=" * 72)
        log(f"PIPELINE FATAL: {type(exc).__name__}: {exc}")
        log("=" * 72)
        log("No training artifacts were modified.")
        try:
            _write_runtime_compat_report()
        except Exception:
            pass
        tb = traceback.format_exc().splitlines()[-6:]
        for line in tb:
            log(f"  {line}")
        return 1


if __name__ == "__main__":
    _code = _safe_main()
    # No sys.exit() — prevents IPython's internal error cascade.
    print(f"[Shadow-Ronin] exit_code={_code}", flush=True)