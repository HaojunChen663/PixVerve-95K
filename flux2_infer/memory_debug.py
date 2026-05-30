import os
import torch


_ENABLED = os.environ.get("DIFFSYNTH_MEMORY_DEBUG", "0") == "1"
_MAX_STEPS = int(os.environ.get("DIFFSYNTH_MEMORY_DEBUG_STEPS", "3"))
_STEP = 0


def is_enabled():
    return _ENABLED


def get_step():
    return _STEP


def increment_step():
    global _STEP
    _STEP += 1
    return _STEP


def should_debug_this_step():
    return _ENABLED and _STEP < _MAX_STEPS


def _memory_text():
    if not torch.cuda.is_available():
        return "cuda=N/A"
    device = torch.cuda.current_device()
    return (
        f"alloc={torch.cuda.memory_allocated(device) / 1024 / 1024:.1f}MB "
        f"reserved={torch.cuda.memory_reserved(device) / 1024 / 1024:.1f}MB "
        f"peak={torch.cuda.max_memory_allocated(device) / 1024 / 1024:.1f}MB"
    )


def log_memory(tag, extra_info=""):
    if not should_debug_this_step():
        return
    suffix = f" | {extra_info}" if extra_info else ""
    print(f"[MEM_DEBUG][step={_STEP}] {tag}: {_memory_text()}{suffix}", flush=True)


def log_tensor(tag, name, tensor):
    if not should_debug_this_step() or tensor is None:
        return
    size_mb = tensor.element_size() * tensor.nelement() / 1024 / 1024
    print(
        f"[MEM_DEBUG][step={_STEP}] {tag} | {name}: "
        f"shape={list(tensor.shape)} dtype={tensor.dtype} device={tensor.device} size={size_mb:.2f}MB",
        flush=True,
    )


def log_tensors(tag, **named_tensors):
    for name, tensor in named_tensors.items():
        log_tensor(tag, name, tensor)
