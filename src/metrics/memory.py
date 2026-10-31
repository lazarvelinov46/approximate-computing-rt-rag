"""Per-device GPU memory for runs whose model may span several cards.

The reference arms (Qwen2.5-3B, Llama-3.2-3B) are pinned to cuda:0, and every
memory number they report is device 0's. Qwen2.5-7B is split across two T4s,
so its peaks are per card. Those numbers are NOT comparable to the single-GPU
arms: each card holds part of the weights, the KV cache is spread over both,
and activations cross between them. `comparable_to_single_gpu` travels with
every snapshot so a reader of the JSON cannot miss it.

Quality sweeps record memory as a diagnostic only. Timing and energy are
measured on a dedicated GPU later and are not read from these runs.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch

GIB = 2 ** 30


def model_devices(model) -> List[int]:
    """CUDA indices holding any of the model's parameters, ascending."""
    return sorted({p.device.index for p in model.parameters()
                   if p.device.type == "cuda"})


def reset_peaks(devices: Optional[Iterable[int]] = None) -> None:
    """Reset peak stats on every listed device (default: all visible)."""
    for d in (range(torch.cuda.device_count()) if devices is None else devices):
        torch.cuda.reset_peak_memory_stats(d)


def snapshot(devices: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Allocated / peak / reserved / free GiB per device, plus the peak sum.

    Call reset_peaks() on the same devices first, or the peak covers the whole
    session since the last reset.
    """
    devs = list(range(torch.cuda.device_count()) if devices is None else devices)
    rec: Dict[str, Any] = {}
    for d in devs:
        free, total = torch.cuda.mem_get_info(d)
        rec[f"cuda:{d}"] = {
            "allocated": round(torch.cuda.memory_allocated(d) / GIB, 3),
            "peak": round(torch.cuda.max_memory_allocated(d) / GIB, 3),
            "reserved": round(torch.cuda.memory_reserved(d) / GIB, 3),
            "free": round(free / GIB, 3),
            "total": round(total / GIB, 3),
        }
    rec["peak_sum"] = round(sum(rec[f"cuda:{d}"]["peak"] for d in devs), 3)
    rec["n_devices"] = len(devs)
    rec["comparable_to_single_gpu"] = len(devs) == 1
    return rec


def fmt(rec: Dict[str, Any]) -> str:
    """One line for notebook output."""
    cards = "  ".join(f"{k} peak {v['peak']:.2f} free {v['free']:.2f}"
                      for k, v in rec.items() if k.startswith("cuda:"))
    tag = "" if rec["comparable_to_single_gpu"] else "  [split: not comparable]"
    return f"{cards}  | sum {rec['peak_sum']:.2f} GiB{tag}"