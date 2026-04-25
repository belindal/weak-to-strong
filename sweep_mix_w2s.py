"""
Sweep: label mixing for weak-to-strong generalization using train_simple.

Runs transfer jobs concurrently with per-model-size GPU memory limits:
  gpt2-xl:     1 job at a time
  gpt2-large:  2 jobs at a time
  gpt2-medium: 4 jobs at a time
  gpt2:        4 jobs at a time

Jobs within each size class are dispatched greedily (largest-first overall),
so GPU slots are never idle while work remains.

Usage
-----
# Defaults: boolq, gpt2->gpt2-xl, random+weak_active, seed 0
  python sweep_mix_w2s.py

# Sweep over weak and strong model sizes
  python sweep_mix_w2s.py \
    --weak_model_sizes gpt2,gpt2-medium \
    --strong_model_sizes gpt2-large,gpt2-xl \
    --ds_names boolq,sciq \
    --seeds 0,1,2

# Skip weak model training (labels already generated)
  python sweep_mix_w2s.py --train_weak False

# Include strong-model active learning (least_confidence) alongside standard mixing
  python sweep_mix_w2s.py --al_strategies least_confidence

# AL only, no standard mixing
  python sweep_mix_w2s.py --mix_selections "" --al_strategies least_confidence --mix_ratio 0.10
"""

import csv
import itertools
import json
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Any, Dict, Sequence, Union

import fire

from train_simple import MODELS_DICT, get_config_foldername
from weak_to_strong.datasets import VALID_DATASETS


# Max concurrent jobs per model size (GPU memory budget).
SIZE_CONCURRENCY: Dict[str, int] = {
    "gpt2-xl": 1,
    "gpt2-large": 2,
    "gpt2-medium": 4,
    "gpt2": 4,
}
# Largest first — minimizes makespan by starting long jobs early.
_SIZE_ORDER = list(SIZE_CONCURRENCY.keys())


def _size_rank(model_size: str) -> int:
    return _SIZE_ORDER.index(model_size) if model_size in _SIZE_ORDER else len(_SIZE_ORDER)


def _to_list(x, sep=","):
    if isinstance(x, str):
        return [s.strip() for s in x.split(sep) if s.strip()]
    return list(x)


def _build_cmd(**kwargs) -> list:
    """Build a train_simple.py subprocess command from keyword arguments."""
    cmd = [sys.executable, "train_simple.py"]
    for k, v in kwargs.items():
        if v is not None:
            cmd += [f"--{k}", str(v)]
    return cmd


def _run_job(cmd: list, semaphore: Semaphore, label: str) -> int:
    with semaphore:
        print(f"  START  {label}")
        proc = subprocess.run(cmd)
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {status} {label}")
        return proc.returncode


def _get_save_path(
    results_folder: str,
    sweep_subfolder: str,
    model_size: str,
    ds_name: str,
    loss: str,
    n_docs: int,
    n_test_docs: int,
    epochs: int,
    batch_size: int,
    lr_schedule: str,
    eval_every: int,
    seed: int,
    weak_model_size: str | None = None,
    mix_ratio: float | None = None,
    mix_selection: str | None = None,
    al_strategy: str | None = None,
    n_al_rounds: int = 1,
) -> str:
    """Replicate train_simple's config → folder-name logic to find the results JSON."""
    mc = MODELS_DICT[model_size]
    config: Dict[str, Any] = {
        "batch_size": batch_size,
        "max_ctx": 1024,
        "ds_name": ds_name,
        "loss": loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": mc.default_lr,
        "optim": mc.default_optimizer,
        "epochs": epochs,
        "seed": seed,
        "train_with_dropout": False,
        "linear_probe": False,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
    }
    if mix_ratio is not None:
        config["mix_ratio"] = mix_ratio
        if al_strategy is not None:
            config["al_strategy"] = al_strategy
            if n_al_rounds > 1:
                config["n_al_rounds"] = n_al_rounds
        elif mix_selection is not None:
            config["mix_selection"] = mix_selection
    if weak_model_size is not None:
        config["weak_model_size"] = weak_model_size
    return os.path.join(results_folder, sweep_subfolder, get_config_foldername(config))


def _read_accuracy(save_path: str) -> float | None:
    summary = os.path.join(save_path, "results_summary.json")
    if not os.path.exists(summary):
        return None
    with open(summary) as f:
        return json.load(f).get("accuracy")


def _run_phase(jobs: list, semaphores: Dict[str, Semaphore], default_sem: Semaphore) -> None:
    """Run a list of jobs concurrently, respecting per-size semaphores. Raises on any failure."""
    if not jobs:
        return
    jobs = sorted(jobs, key=lambda j: _size_rank(j["model_size"]))
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futs = {
            pool.submit(
                _run_job,
                j["cmd"],
                semaphores.get(j["model_size"], default_sem),
                j["label"],
            ): j
            for j in jobs
        }
        for f in as_completed(futs):
            rc = f.result()
            if rc != 0:
                raise RuntimeError(f"Job failed (rc={rc}): {futs[f]['label']}")


def sweep(
    ds_names: Union[str, Sequence[str]] = "boolq",
    weak_model_sizes: Union[str, Sequence[str]] = "gpt2",
    strong_model_sizes: Union[str, Sequence[str]] = "gpt2-xl",
    loss: str = "xent",
    seeds: Union[int, Sequence[int]] = 0,
    mix_ratio: float = 0.25,
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    epochs: int = 2,
    batch_size: int = 32,
    minibatch_size_per_device: int = 32,
    max_ctx: int = 1024,
    results_folder: str = "/workspace/weak-to-strong/results",
    sweep_subfolder: str = "default",
    lr_schedule: str = "cosine_anneal",
    eval_every: int = 1000000,
    force_retrain: bool = False,
    mix_selections: Union[str, Sequence[str]] = "random,weak_active",
    # AL strategies to sweep; empty string skips AL entirely.
    # e.g. "least_confidence" or "least_confidence,random"
    al_strategies: Union[str, Sequence[str]] = "",
    # Number of AL rounds to sweep; can be a single value or comma-separated list.
    n_al_rounds: Union[int, Sequence[int]] = 1,
    # Set False to skip weak model training (e.g. labels already generated).
    train_weak: bool = True,
    # Set False to skip Phase 2 (mix + baseline runs, e.g. already completed).
    train_transfer: bool = True,
):
    ds_names = _to_list(ds_names)
    weak_model_sizes = _to_list(weak_model_sizes)
    strong_model_sizes = _to_list(strong_model_sizes)
    seeds = [seeds] if isinstance(seeds, int) else list(seeds)
    selections = _to_list(mix_selections)
    al_strats = _to_list(al_strategies)
    n_al_rounds_list = [n_al_rounds] if isinstance(n_al_rounds, int) else [int(x) for x in n_al_rounds]

    for ds in ds_names:
        assert ds in VALID_DATASETS, f"Unknown dataset {ds!r}; valid: {list(VALID_DATASETS)}"
    for m in weak_model_sizes + strong_model_sizes:
        assert m in MODELS_DICT, f"Unknown model {m!r}; valid: {list(MODELS_DICT)}"

    configs = [
        (ds, weak_size, strong_size, seed)
        for ds, weak_size, strong_size, seed
        in itertools.product(ds_names, weak_model_sizes, strong_model_sizes, seeds)
        if _size_rank(weak_size) >= _size_rank(strong_size)
    ]
    n_mix = len(configs) * len(selections)
    n_al = len(configs) * len(al_strats) * len(n_al_rounds_list)
    print(
        f"Sweep: {len(configs)} configs × {len(selections)} mix selections = {n_mix} mix runs"
        + (f", × {len(al_strats)} AL strategies = {n_al} AL runs" if al_strats else "") + "\n"
        f"  mix_ratio={mix_ratio:.0%}, selections={selections}, seeds={seeds}\n"
        + (f"  al_strategies={al_strats}\n" if al_strats else "")
        + f"  concurrency: {SIZE_CONCURRENCY}"
    )

    os.makedirs(results_folder, exist_ok=True)
    semaphores = {size: Semaphore(n) for size, n in SIZE_CONCURRENCY.items()}
    default_sem = Semaphore(1)

    common: Dict[str, Any] = dict(
        loss=loss,
        n_docs=n_docs,
        n_test_docs=n_test_docs,
        epochs=epochs,
        batch_size=batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        max_ctx=max_ctx,
        results_folder=results_folder,
        sweep_subfolder=sweep_subfolder,
        lr_schedule=lr_schedule,
        eval_every=eval_every,
        force_retrain=force_retrain,
    )

    # ── Phase 1: generate weak labels ─────────────────────────────────────────
    if train_weak:
        seen: set = set()
        weak_jobs = []
        for ds, weak_size, _, seed in configs:
            key = (ds, weak_size, seed)
            if key in seen:
                continue
            seen.add(key)
            weak_jobs.append({
                "model_size": weak_size,
                "label": f"weak {ds} {weak_size} s{seed}",
                "cmd": _build_cmd(model_size=weak_size, ds_name=ds, seed=seed, **common),
            })
        print(f"\nPhase 1: {len(weak_jobs)} weak-model runs")
        _run_phase(weak_jobs, semaphores, default_sem)

    # ── Phase 2: standard mix runs + W2S baseline (needed as AL checkpoint) ───
    transfer_jobs = []
    for ds, weak_size, strong_size, seed in configs:
        for selection in selections:
            transfer_jobs.append({
                "model_size": strong_size,
                "weak_size": weak_size,
                "ds": ds,
                "seed": seed,
                "selection": selection,
                "al_strategy": None,
                "label": f"transfer {ds} {weak_size}→{strong_size} {selection} s{seed}",
                "cmd": _build_cmd(
                    model_size=strong_size,
                    weak_model_size=weak_size,
                    ds_name=ds,
                    seed=seed,
                    mix_ratio=mix_ratio,
                    mix_selection=selection,
                    **common,
                ),
            })

    # W2S baseline (no mixing) is the checkpoint AL scoring loads from.
    baseline_jobs = []
    if al_strats:
        seen_baseline: set = set()
        for ds, weak_size, strong_size, seed in configs:
            key = (ds, weak_size, strong_size, seed)
            if key in seen_baseline:
                continue
            seen_baseline.add(key)
            baseline_jobs.append({
                "model_size": strong_size,
                "label": f"baseline {ds} {weak_size}→{strong_size} s{seed}",
                "cmd": _build_cmd(
                    model_size=strong_size,
                    weak_model_size=weak_size,
                    ds_name=ds,
                    seed=seed,
                    **common,
                ),
            })

    phase2_jobs = transfer_jobs + baseline_jobs
    print(f"\nPhase 2: {len(phase2_jobs)} runs ({len(transfer_jobs)} mix, {len(baseline_jobs)} baseline)")
    if train_transfer:
        _run_phase(phase2_jobs, semaphores, default_sem)
    else:
        print("  Skipping (train_transfer=False)")

    # ── Phase 3: AL (strong_active) runs ──────────────────────────────────────
    al_jobs = []
    for ds, weak_size, strong_size, seed in configs:
        for al_strat in al_strats:
            for n_rounds in n_al_rounds_list:
                rounds_tag = f" r{n_rounds}" if n_rounds > 1 else ""
                al_jobs.append({
                    "model_size": strong_size,
                    "weak_size": weak_size,
                    "ds": ds,
                    "seed": seed,
                    "selection": f"strong_active/{al_strat}",
                    "al_strategy": al_strat,
                    "n_al_rounds": n_rounds,
                    "label": f"AL {ds} {weak_size}→{strong_size} {al_strat}{rounds_tag} s{seed}",
                    "cmd": _build_cmd(
                        model_size=strong_size,
                        weak_model_size=weak_size,
                        ds_name=ds,
                        seed=seed,
                        mix_ratio=mix_ratio,
                        al_strategy=al_strat,
                        n_al_rounds=n_rounds if n_rounds > 1 else None,
                        **common,
                    ),
                })

    if al_jobs:
        print(f"\nPhase 3: {len(al_jobs)} AL runs")
        _run_phase(al_jobs, semaphores, default_sem)

    # ── Collect results ────────────────────────────────────────────────────────
    save_path_kwargs = dict(
        results_folder=results_folder,
        sweep_subfolder=sweep_subfolder,
        loss=loss,
        n_docs=n_docs,
        n_test_docs=n_test_docs,
        epochs=epochs,
        batch_size=batch_size,
        lr_schedule=lr_schedule,
        eval_every=eval_every,
    )
    rows = []
    for j in transfer_jobs + al_jobs:
        save_path = _get_save_path(
            model_size=j["model_size"],
            ds_name=j["ds"],
            seed=j["seed"],
            weak_model_size=j["weak_size"],
            mix_ratio=mix_ratio,
            mix_selection=j["selection"] if j["al_strategy"] is None else None,
            al_strategy=j["al_strategy"],
            n_al_rounds=j.get("n_al_rounds", 1),
            **save_path_kwargs,
        )
        weak_save_path = _get_save_path(
            model_size=j["weak_size"],
            ds_name=j["ds"],
            seed=j["seed"],
            **save_path_kwargs,
        )
        rows.append({
            "ds_name": j["ds"],
            "weak_model_size": j["weak_size"],
            "model_size": j["model_size"],
            "seed": j["seed"],
            "mix_ratio": mix_ratio,
            "selection": j["selection"],
            "n_al_rounds": j.get("n_al_rounds", 1),
            "weak_acc": _read_accuracy(weak_save_path),
            "transfer_acc": _read_accuracy(save_path),
        })

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"RESULTS  (mix_ratio={mix_ratio:.0%})")
    print(f"{'='*80}")

    col_w = 48
    header = f"{'Config':<{col_w}} {'selection':<10} {'weak':>7} {'transfer':>9}"
    print(header)
    print("-" * len(header))

    def _fmt(v, width=9):
        return f"{v:.4f}".rjust(width) if v is not None else "N/A".rjust(width)

    for r in rows:
        config_str = f"{r['ds_name']} {r['weak_model_size']}→{r['model_size']} s{r['seed']}"
        print(
            f"{config_str:<{col_w}} {r['selection']:<10}"
            f" {_fmt(r['weak_acc'], 7)} {_fmt(r['transfer_acc'])}"
        )

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_json = os.path.join(results_folder, "sweep_mix_results.json")
    out_csv = os.path.join(results_folder, "sweep_mix_results.csv")

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nJSON: {out_json}")
    print(f"CSV:  {out_csv}")
    return rows


if __name__ == "__main__":
    fire.Fire(sweep)
