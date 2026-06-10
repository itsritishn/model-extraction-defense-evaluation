"""
evaluate.py
===========
Full evaluation pipeline for model extraction attack experiments.

Computes all metrics, saves results to CSV, and generates publication-
quality charts (bar, scatter, line, heatmap) with seaborn + matplotlib.

Metrics computed for every (attack × defense × budget) combination:
  • victim_accuracy    – victim's top-1 accuracy on CIFAR-10 test
  • substitute_accuracy – substitute's top-1 accuracy
  • fidelity           – agreement rate between victim & substitute
  • protection_score   – 1 − (substitute_accuracy / victim_accuracy)
  • utility_cost       – victim accuracy drop caused by the defense
  • query_efficiency   – fidelity per 1 000 queries
  • attack_roi         – substitute_accuracy / query_budget
  • defense_latency_ms – wall-clock overhead of the defense function

Author  : MSc Advanced Computer Science (Data Analytics) Dissertation
Python  : 3.9.25 (strict – no 3.10+ features)
"""

import csv
import gc
import logging
import os
import time
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from attack import ATTACK_REGISTRY, run_attack
from data_loader import get_data_loaders
from defenses import get_all_defenses
from threat_model import AttackerKnowledge, QueryBudget, ThreatConfig
from victim import DEVICE, VictimModel, build_victim_model
from utils import AMP_DEVICE, AMP_ENABLED


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

RESULTS_DIR: str = "experiments/results"
CHARTS_DIR: str = "experiments/charts"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


def _empty_cache() -> None:
    """Release cached device memory, whichever backend is active.

    ``torch.cuda.empty_cache()`` is a no-op on Apple-Silicon (MPS) and CPU, so
    long sweeps on those backends accumulate memory and can exhaust system RAM.
    This helper clears the cache for the active backend instead.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    mps = getattr(torch, "mps", None)
    backends_mps = getattr(torch.backends, "mps", None)
    if mps is not None and backends_mps is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


@torch.no_grad()
def compute_accuracy(model: torch.nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
            outputs = model(images)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def precompute_victim_outputs(
    victim: VictimModel,
    data_loader: DataLoader,
) -> "tuple":
    """Compute the frozen victim's clean probabilities + labels ONCE.

    The victim never changes during evaluation, so its predictions on the test
    set are identical for every (attack x defense x budget) combination.
    Caching them here and reusing the cache in the per-combo metrics removes
    the single largest cost in the pipeline: hundreds of redundant full-test
    forward passes through ResNet-50 / EfficientNet.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``(victim_probs, labels)`` on CPU, in test-loader order.
    """
    all_probs = []
    all_labels = []
    for images, labels in data_loader:
        all_probs.append(victim.query(images))   # clean, no defense; CPU tensor
        all_labels.append(labels)
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def compute_defended_accuracy(
    victim_probs: torch.Tensor,
    data_loader: DataLoader,
    defense_fn: Optional[object] = None,
) -> float:
    """Top-1 accuracy of the *defended* victim as seen by a legitimate user.

    Uses the cached clean victim probabilities (``victim_probs``) and replays
    the defense over them batch-by-batch in test-loader order — identical to
    querying the live victim, but with no victim forward passes. This is what
    makes ``utility_cost`` a real measured quantity. Stateful defenses are
    reset before and after so the measurement does not leak counter state.

    Returns
    -------
    float
        Defended accuracy in percent (0-100).
    """
    if defense_fn is not None and hasattr(defense_fn, "reset"):
        defense_fn.reset()

    correct = 0
    total = 0
    offset = 0
    for _, labels in data_loader:
        bs = labels.size(0)
        batch = victim_probs[offset:offset + bs].clone()
        offset += bs
        if defense_fn is not None:
            batch = defense_fn(batch)
        preds = batch.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

    if defense_fn is not None and hasattr(defense_fn, "reset"):
        defense_fn.reset()

    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_fidelity(
    substitute: torch.nn.Module,
    data_loader: DataLoader,
    victim_preds: torch.Tensor,
) -> float:
    """Agreement (%) between the substitute and the (cached) victim predictions.

    ``victim_preds`` is the precomputed argmax of the victim's clean outputs,
    so no victim forward pass is needed here — only the cheap substitute runs.
    """
    substitute.eval()
    agree = 0
    total = 0
    offset = 0
    for images, _ in data_loader:
        bs = images.size(0)
        images_dev = images.to(DEVICE, non_blocking=True)
        with autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
            sub_logits = substitute(images_dev)
        sub_preds = sub_logits.argmax(dim=1).cpu()
        agree += (victim_preds[offset:offset + bs] == sub_preds).sum().item()
        offset += bs
        total += bs
    return 100.0 * agree / total if total > 0 else 0.0


def compute_defense_latency(
    defense_fn: Optional[object],
    dummy_probs: torch.Tensor,
    repeats: int = 100,
) -> float:
    if defense_fn is None:
        return 0.0
    for _ in range(5):
        defense_fn(dummy_probs)
    t0 = time.perf_counter()
    for _ in range(repeats):
        defense_fn(dummy_probs)
    t1 = time.perf_counter()
    return round((t1 - t0) / repeats * 1000.0, 3)


def compute_all_metrics(
    victim: VictimModel,
    victim_model_raw: torch.nn.Module,
    substitute: torch.nn.Module,
    test_loader: DataLoader,
    query_budget: int,
    defense_fn: Optional[object] = None,
    victim_acc_baseline: Optional[float] = None,
    victim_test_probs: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    logger.info("Computing evaluation metrics …")

    # Use cached victim outputs when available (avoids re-running the victim).
    if victim_test_probs is None:
        victim_test_probs, _ = precompute_victim_outputs(victim, test_loader)
    victim_preds = victim_test_probs.argmax(dim=1)

    victim_acc = (
        victim_acc_baseline
        if victim_acc_baseline is not None
        else compute_accuracy(victim_model_raw, test_loader)
    )
    sub_acc = compute_accuracy(substitute, test_loader)
    fidelity = compute_fidelity(substitute, test_loader, victim_preds)

    protection = 1.0 - (sub_acc / victim_acc) if victim_acc > 0 else 1.0
    protection = max(0.0, min(1.0, protection))

    # Utility cost = accuracy a legitimate user loses because of the defense,
    # replayed over the cached clean victim outputs.
    if defense_fn is not None and victim_acc_baseline is not None:
        defended_acc = compute_defended_accuracy(victim_test_probs, test_loader, defense_fn)
        utility_cost = max(0.0, victim_acc_baseline - defended_acc)
    else:
        utility_cost = 0.0

    query_eff = (fidelity / query_budget * 1000.0) if query_budget > 0 else 0.0
    attack_roi = sub_acc / query_budget if query_budget > 0 else 0.0

    dummy = torch.softmax(torch.randn(64, 10), dim=1)
    latency_ms = compute_defense_latency(defense_fn, dummy)

    metrics = {
        "victim_accuracy": round(victim_acc, 2),
        "substitute_accuracy": round(sub_acc, 2),
        "fidelity": round(fidelity, 2),
        "protection_score": round(protection, 4),
        "utility_cost": round(utility_cost, 2),
        "query_efficiency": round(query_eff, 4),
        "attack_roi": round(attack_roi, 6),
        "defense_latency_ms": latency_ms,
    }
    logger.info("Metrics: %s", metrics)
    return metrics


def save_results_csv(rows: List[Dict], filename: str = "experiment_results.csv") -> str:
    path = os.path.join(RESULTS_DIR, filename)
    if len(rows) == 0:
        logger.warning("No results to save.")
        return path

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fixed_cols = [
        "attack", "defense", "budget", "substitute_arch", "victim_arch",
        "victim_accuracy", "substitute_accuracy", "fidelity", "protection_score",
        "utility_cost", "query_efficiency", "attack_roi", "defense_latency_ms",
    ]
    meta_cols = sorted([k for k in all_keys if k.startswith("meta_")])
    fieldnames = [c for c in fixed_cols if c in all_keys] + meta_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Results saved → %s  (%d rows)", path, len(rows))
    return path


def load_results_csv(filename: str = "experiment_results.csv") -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, filename)
    df = pd.read_csv(path)
    logger.info("Loaded results ← %s  (%d rows)", path, len(df))
    return df


def _save_fig(fig: plt.Figure, name: str) -> str:
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Chart saved → %s", path)
    return path


def plot_fidelity_per_defense(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby("defense")["fidelity"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="defense", y="fidelity", order=order,
                hue="defense", legend=False, palette="viridis", ax=ax)
    ax.set_title("Attack Fidelity With and Without Defences", fontsize=14)
    ax.set_xlabel("Defense")
    ax.set_ylabel("Fidelity (%)")
    ax.tick_params(axis="x", rotation=45)
    return _save_fig(fig, "fidelity_comparison.png")


def plot_protection_per_defense(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby("defense")["protection_score"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="defense", y="protection_score", order=order,
                hue="defense", legend=False, palette="magma", ax=ax)
    ax.set_title("Protection Score Per Defence", fontsize=14)
    ax.set_xlabel("Defense")
    ax.set_ylabel("Protection Score")
    ax.tick_params(axis="x", rotation=45)
    return _save_fig(fig, "protection_scores.png")


def plot_security_vs_utility(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="utility_cost",
        y="protection_score",
        hue="defense",
        style="attack",
        s=120,
        ax=ax,
    )
    ax.set_title("Security vs Utility Tradeoff", fontsize=14)
    ax.set_xlabel("Utility Cost (accuracy drop %)")
    ax.set_ylabel("Protection Score")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    return _save_fig(fig, "security_utility_tradeoff.png")


def plot_fidelity_vs_budget(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    for attack_name in df["attack"].unique():
        subset = df[df["attack"] == attack_name]
        means = subset.groupby("budget")["fidelity"].mean().sort_index()
        ax.plot(means.index, means.values, marker="o", label=attack_name)
    ax.set_title("Attack Success vs Query Budget", fontsize=14)
    ax.set_xlabel("Query Budget")
    ax.set_ylabel("Fidelity (%)")
    ax.legend()
    ax.set_xscale("log")
    return _save_fig(fig, "fidelity_vs_query_budget.png")


def plot_heatmap(
    df: pd.DataFrame,
    metric: str = "fidelity",
    filename: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    pivot = df.pivot_table(values=metric, index="defense", columns="attack", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(title if title is not None else "Heatmap: {0} (defense × attack)".format(metric), fontsize=14)
    return _save_fig(fig, filename if filename is not None else "heatmap_{0}.png".format(metric))


def generate_all_charts(df: pd.DataFrame) -> List[str]:
    paths = [
        plot_fidelity_per_defense(df),
        plot_protection_per_defense(df),
        plot_security_vs_utility(df),
        plot_fidelity_vs_budget(df),
        plot_heatmap(
            df,
            metric="fidelity",
            filename="attack_defense_heatmap.png",
            title="Fidelity Heatmap: Attack vs Defence",
        ),
    ]
    logger.info("All charts generated (%d files)", len(paths))
    return paths


def run_full_evaluation(
    victim_arch: str = "resnet50",
    victim_checkpoint: Optional[str] = None,
    substitute_archs: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    attack_names: Optional[List[str]] = None,
    defense_names: Optional[List[str]] = None,
    sub_epochs: int = 15,
    batch_size: int = 64,
    data_dir: str = "./data/cifar10",
    results_filename: str = "experiment_results.csv",
    skip_charts: bool = False,
) -> pd.DataFrame:
    if substitute_archs is None:
        substitute_archs = ["small_cnn", "mobilenetv3_small"]
    if budgets is None:
        # Default to the budgets used in the documented evaluation scope (1000, 5000).
        budgets = [QueryBudget.LOW.value, QueryBudget.MEDIUM.value]
    if attack_names is None:
        attack_names = list(ATTACK_REGISTRY.keys())
    if defense_names is None:
        defense_names = [name for name, _ in get_all_defenses()]

    logger.info("=" * 70)
    logger.info("FULL EVALUATION  |  victim=%s", victim_arch)
    logger.info("=" * 70)

    # When a checkpoint is available, the full fine-tuned weights are loaded
    # below, so there is no need to download ImageNet weights (keeps the run
    # fully offline and faster).
    have_ckpt = victim_checkpoint is not None and os.path.isfile(victim_checkpoint)
    victim_model_raw = build_victim_model(victim_arch, pretrained=not have_ckpt)
    if have_ckpt:
        state = torch.load(victim_checkpoint, map_location=DEVICE, weights_only=True)
        victim_model_raw.load_state_dict(state)
        logger.info("Loaded victim checkpoint: %s", victim_checkpoint)

    victim = VictimModel(victim_model_raw)

    loaders = get_data_loaders(batch_size=batch_size, data_dir=data_dir)
    test_loader = loaders["test"]

    victim_acc_baseline = compute_accuracy(victim_model_raw, test_loader)
    logger.info("Victim baseline accuracy: %.2f%%", victim_acc_baseline)

    # Precompute the frozen victim's clean test-set outputs ONCE and reuse them
    # for fidelity + utility-cost in every combo (the big GPU-hour saver).
    victim_test_probs, _ = precompute_victim_outputs(victim, test_loader)
    logger.info("Cached victim test outputs: %s", tuple(victim_test_probs.shape))

    all_defenses_dict = dict(get_all_defenses())

    all_rows = []
    total_combos = len(attack_names) * len(defense_names) * len(budgets) * len(substitute_archs)
    logger.info("Total experiment combinations: %d", total_combos)

    combo_idx = 0
    for attack_name in attack_names:
        for defense_name in defense_names:
            defense_fn = all_defenses_dict.get(defense_name)
            if hasattr(defense_fn, "reset"):
                defense_fn.reset()

            for budget in budgets:
                for sub_arch in substitute_archs:
                    combo_idx += 1
                    substitute = None
                    logger.info(
                        "─── Combo %d/%d  |  attack=%s  defense=%s  budget=%d  sub=%s ───",
                        combo_idx,
                        total_combos,
                        attack_name,
                        defense_name,
                        budget,
                        sub_arch,
                    )

                    config = ThreatConfig(
                        budget=budget,
                        knowledge=AttackerKnowledge.LEVEL_2,
                        substitute_arch=sub_arch,
                        victim_arch=victim_arch,
                        batch_size=batch_size,
                    )

                    try:
                        substitute, meta = run_attack(
                            attack_name=attack_name,
                            victim=victim,
                            config=config,
                            defense_fn=defense_fn,
                            sub_epochs=sub_epochs,
                        )

                        metrics = compute_all_metrics(
                            victim=victim,
                            victim_model_raw=victim_model_raw,
                            substitute=substitute,
                            test_loader=test_loader,
                            query_budget=budget,
                            defense_fn=defense_fn,
                            victim_acc_baseline=victim_acc_baseline,
                            victim_test_probs=victim_test_probs,
                        )

                        row = {
                            "attack": attack_name,
                            "defense": defense_name,
                            "budget": budget,
                            "substitute_arch": sub_arch,
                            "victim_arch": victim_arch,
                            **metrics,
                            **{"meta_{0}".format(k): v for k, v in meta.items()},
                        }
                        all_rows.append(row)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error("OOM for combo %d — skipping. %s", combo_idx, e)
                            _empty_cache()
                            continue
                        raise
                    finally:
                        # Free per-combo memory so long sweeps stay within RAM
                        # (critical on MPS/CPU where caches are not auto-released).
                        if substitute is not None:
                            del substitute
                        _empty_cache()
                        gc.collect()

                    if hasattr(defense_fn, "reset"):
                        defense_fn.reset()

    csv_path = save_results_csv(all_rows, filename=results_filename)
    df = pd.DataFrame(all_rows)

    if len(df) > 0:
        if not skip_charts:
            generate_all_charts(df)
        _print_summary_table(df)
    else:
        logger.warning("No results collected — charts skipped.")

    logger.info("Evaluation complete  |  results → %s", csv_path)
    return df


def _print_summary_table(df: pd.DataFrame) -> None:
    summary = df.groupby(["defense", "attack"]).agg({
        "substitute_accuracy": "mean",
        "fidelity": "mean",
        "protection_score": "mean",
        "utility_cost": "mean",
        "defense_latency_ms": "mean",
    }).round(2)

    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 90)
    logger.info(
        "%-25s %-18s %8s %8s %10s %10s %10s",
        "Defense",
        "Attack",
        "Sub_Acc",
        "Fidelity",
        "Protect",
        "Util_Cost",
        "Latency",
    )
    logger.info("-" * 90)
    for (defense, attack), row in summary.iterrows():
        logger.info(
            "%-25s %-18s %7.2f%% %7.2f%% %9.4f %9.2f%% %8.3fms",
            defense,
            attack,
            row["substitute_accuracy"],
            row["fidelity"],
            row["protection_score"],
            row["utility_cost"],
            row["defense_latency_ms"],
        )
    logger.info("=" * 90)


def print_final_conclusion(df_no_def: pd.DataFrame, df_with_def: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FINAL CONCLUSION")
    print("=" * 60)

    if len(df_no_def) > 0:
        best_attack = df_no_def.groupby("attack")["fidelity"].mean().idxmax()
        best_fid = df_no_def.groupby("attack")["fidelity"].mean().max()
        print("  Best attack strategy:      {0} (avg fidelity: {1:.2f}%)".format(best_attack, best_fid))

    if len(df_with_def) > 0:
        best_def = df_with_def.groupby("defense")["protection_score"].mean().idxmax()
        best_prot = df_with_def.groupby("defense")["protection_score"].mean().max()
        print("  Most effective defence:     {0} (avg protection: {1:.4f})".format(best_def, best_prot))

        least_def = df_with_def.groupby("defense")["utility_cost"].mean().idxmin()
        least_val = df_with_def.groupby("defense")["utility_cost"].mean().min()
        print("  Least utility cost defence: {0} (avg cost: {1:.2f}%)".format(least_def, least_val))

    if len(df_no_def) > 0:
        avg_fid = df_no_def["fidelity"].mean()
        feasible = "YES" if avg_fid > 50.0 else "NO"
        print(
            "  Overall: Was extraction feasible? {0} (avg fidelity without defence: {1:.2f}%)".format(
                feasible,
                avg_fid,
            )
        )

    if len(df_no_def) > 0 and len(df_with_def) > 0:
        avg_no = df_no_def["fidelity"].mean()
        avg_with = df_with_def["fidelity"].mean()
        reduction = avg_no - avg_with
        worked = "YES" if reduction > 2.0 else "NO"
        print("  Overall: Did defences work?      {0} (fidelity reduced by {1:.2f}%)".format(worked, reduction))

    print("=" * 60 + "\n")


if __name__ == "__main__":
    VICTIMS = [
        ("resnet50", "models/victim_resnet50.pth"),
    ]
    # Single victim: ResNet-50 carries the full attack-defence matrix.
    PRIMARY_VICTIM = "resnet50"
    BUDGETS = [1000, 5000]
    ATTACKS = ["random_query", "knockoff_nets", "active_learning"]
    SUBSTITUTES = ["small_cnn", "mobilenetv3_small"]

    logger.info("=" * 60)
    logger.info("PHASE 1: Checking victim model")
    logger.info("=" * 60)

    for victim_arch, victim_checkpoint in VICTIMS:
        if not os.path.isfile(victim_checkpoint):
            logger.error("Victim not found at %s — run: python victim.py", victim_checkpoint)
            raise SystemExit(1)
        logger.info("Victim checkpoint OK: %s", victim_checkpoint)

        logger.info("=" * 60)
        logger.info("PHASE 2: Attacks WITHOUT defence  |  victim=%s", victim_arch)
        logger.info("=" * 60)
        df_no_def = run_full_evaluation(
            victim_arch=victim_arch,
            victim_checkpoint=victim_checkpoint,
            substitute_archs=SUBSTITUTES,
            budgets=BUDGETS,
            attack_names=ATTACKS,
            defense_names=["none"],
            sub_epochs=15,
            results_filename="results_no_defense_{0}.csv".format(victim_arch),
            skip_charts=True,
        )

        logger.info("=" * 60)
        logger.info("PHASE 3: Attacks WITH defences  |  victim=%s", victim_arch)
        logger.info("=" * 60)
        if victim_arch == PRIMARY_VICTIM:
            defense_list = [name for name, _ in get_all_defenses() if name != "none"]
            df_with_def = run_full_evaluation(
                victim_arch=victim_arch,
                victim_checkpoint=victim_checkpoint,
                substitute_archs=SUBSTITUTES,
                budgets=BUDGETS,
                attack_names=ATTACKS,
                defense_names=defense_list,
                sub_epochs=15,
                results_filename="results_with_defense_{0}.csv".format(victim_arch),
                skip_charts=True,
            )
        else:
            logger.info(
                "Secondary victim %s — baseline-only by design; skipping defended phase.",
                victim_arch,
            )
            df_with_def = pd.DataFrame()

        logger.info("=" * 60)
        logger.info("PHASE 4: Evaluation & Comparison  |  victim=%s", victim_arch)
        logger.info("=" * 60)
        df_all = pd.concat([df_no_def, df_with_def], ignore_index=True)
        # Only the primary victim has the full attack-defence matrix, so charts
        # are generated from it. Generating from a baseline-only secondary victim
        # would overwrite the primary victim's charts with incomplete data.
        if victim_arch == PRIMARY_VICTIM and len(df_all) > 0:
            generate_all_charts(df_all)
            _print_summary_table(df_all)
        print_final_conclusion(df_no_def, df_with_def)

    logger.info("Full 4-phase pipeline complete.")
