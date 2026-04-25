import json
import os
import random
import subprocess
from typing import Dict, List, Optional

import fire
import numpy as np
import torch
from datasets import load_from_disk

import weak_to_strong.logger as logger
from weak_to_strong.common import clear_mem, get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset,
                                     tokenize_dataset, mix_labels)
from weak_to_strong.eval import eval_model_acc
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.train import ModelConfig, train_and_save_model

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        # Should use model_parallel on V100s (note: ironically if you have a single V100 it should run,
        # but if you have multiple it won't run without model_parallel because of the overhead of data
        # parallel training).
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "5fde88dff770a7d036847211f5d9d9705f0caa69",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "d4efd21e866b9cb3466cb65b963933f5e98016d1",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-14B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this bf16 support and without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "8be2854218fea9054331e217fd26a06f3fd02004",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "fec78c0e3b3b10dd9f0ce775c34a686a3255a7d1",
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]
MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())

VALID_AL_STRATEGIES = ["least_confidence", "entropy", "margin", "random", "disagreement"]


def _score_examples(
    strong_soft_labels: np.ndarray,
    strategy: str,
    weak_soft_labels: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return per-example selection scores (higher = more worth labeling).

    strong_soft_labels: shape [n, 2], current model's softmax predictions.
    weak_soft_labels:   shape [n, 2], required when strategy='disagreement'.
    """
    if strategy == "least_confidence":
        return 1.0 - np.max(strong_soft_labels, axis=1)
    if strategy == "entropy":
        eps = 1e-10
        return -np.sum(strong_soft_labels * np.log(strong_soft_labels + eps), axis=1)
    if strategy == "margin":
        sorted_p = np.sort(strong_soft_labels, axis=1)[:, ::-1]
        return 1.0 - (sorted_p[:, 0] - sorted_p[:, 1])
    if strategy == "random":
        return np.random.default_rng(seed).random(len(strong_soft_labels))
    if strategy == "disagreement":
        assert weak_soft_labels is not None, "disagreement strategy requires weak_soft_labels"
        return np.abs(weak_soft_labels[:, 1] - strong_soft_labels[:, 1])
    raise ValueError(f"Unknown al_strategy {strategy!r}; choose from {VALID_AL_STRATEGIES}")


def _score_pool(
    model_config: "ModelConfig",
    checkpoint: str,
    ds,
    eval_batch_size: int,
):
    """Load a model from checkpoint and return eval_model_acc results for ds."""
    custom_kwargs = model_config.custom_kwargs or {}
    if model_config.model_parallel:
        model = TransformerWithHead.from_pretrained(
            model_config.name, num_labels=2, device_map="auto", **custom_kwargs
        )
    else:
        model = TransformerWithHead.from_pretrained(
            model_config.name, num_labels=2, **custom_kwargs
        ).to("cuda")
    ckpt_bin = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(ckpt_bin):
        state_dict = torch.load(ckpt_bin, map_location="cpu")
        state_dict = {
            k.replace("transformer.module", "transformer"): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
    scored = eval_model_acc(model, ds, eval_batch_size)
    del model
    clear_mem()
    return scored


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    loss: str = "xent",
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    epochs: int = 2,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[int] = None,
    train_with_dropout: bool = False,
    results_folder: str = "/workspace/weak-to-strong/results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    # Note: you can pass either weak_model_size or weak_labels_path. If you pass
    # weak_model_size, we will guess the path to the weak labels based on the weak
    # model. If you pass weak_labels_path, we will use that path instead.
    # If you pass neither, we will train on ground truth.
    weak_model_size: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "default",
    # Set to a very large value so that by default we don't do any intermediate evals but
    # still do final evals (which requires eval_every to be set to a non-zero, non-None value)
    eval_every: int = 1000000,
    sync_command: Optional[str] = None,
    # Fraction of training examples to replace with strong (ground truth) labels.
    # The remaining (1 - mix_ratio) fraction use weak model labels.
    mix_ratio: Optional[float] = None,
    # How to select which examples get strong labels: "random" or "active".
    # "active": replace the mix_ratio fraction with lowest weak-model confidence.
    mix_selection: str = "random",
    resume_from_checkpoint: Optional[str] = None,
    # When set, scores the training pool with the strong model checkpoint and
    # replaces the top mix_ratio fraction (by uncertainty) with GT labels.
    # Requires both --resume_from_checkpoint and --mix_ratio to be set.
    # Choices: least_confidence | entropy | margin | random
    al_strategy: Optional[str] = None,
    # Number of active learning rounds. Each round reveals mix_ratio/n_al_rounds
    # GT labels, trains, then re-scores the remaining unlabeled pool.
    # Requires al_strategy to be set when > 1.
    n_al_rounds: int = 1,
):
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    if mix_ratio is not None:
        assert 0.0 <= mix_ratio <= 1.0, f"mix_ratio must be in [0, 1], got {mix_ratio}"
    if al_strategy is not None:
        assert al_strategy in VALID_AL_STRATEGIES, f"al_strategy must be one of {VALID_AL_STRATEGIES}"
        assert mix_ratio is not None, "al_strategy requires --mix_ratio to set the GT-label budget"
        assert weak_labels_path is not None or weak_model_size is not None, \
            "al_strategy requires weak labels (--weak_labels_path or --weak_model_size)"
        if al_strategy == "disagreement":
            assert weak_labels_path is not None or weak_model_size is not None, \
                "al_strategy='disagreement' requires weak labels to compare against"
    if n_al_rounds > 1:
        assert al_strategy is not None, "n_al_rounds > 1 requires --al_strategy"

    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    assert (
        weak_model_size is None or weak_labels_path is None
    ), "Can't pass both weak_model_size and weak_labels_path"
    model_config = MODELS_DICT[model_size]

    use_default_lr = False
    if lr is None:
        assert (
            batch_size == 32
        ), "Learning rates were tuned on batch size 32, you probably want to sweep LR if you are tuning batch size"
        lr = model_config.default_lr
        use_default_lr = True

    if optim is None:
        optim = model_config.default_optimizer

    # The commented out terms are the ones that should not change final results
    config = {
        "batch_size": batch_size,
        "max_ctx": max_ctx,
        "ds_name": ds_name,
        "loss": loss,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        # "force_retrain": force_retrain,
        "seed": seed,
        # "minibatch_size_per_device": minibatch_size_per_device,
        "train_with_dropout": train_with_dropout,
        # "results_folder": results_folder,
        "linear_probe": linear_probe,
        "lr_schedule": lr_schedule,
        "eval_every": eval_every,
        # "sweep_subfolder": sweep_subfolder,
    }
    if mix_ratio is not None:
        config["mix_ratio"] = mix_ratio
        if al_strategy is not None:
            config["al_strategy"] = al_strategy
            if n_al_rounds > 1:
                config["n_al_rounds"] = n_al_rounds
        else:
            config["mix_selection"] = mix_selection

    if weak_model_size is not None:
        weak_model_config = config.copy()
        weak_model_config["model_size"] = weak_model_size
        weak_model_config["loss"] = "xent"
        del weak_model_config["mix_ratio"]
        if "mix_selection" in weak_model_config:
            del weak_model_config["mix_selection"]
        if "al_strategy" in weak_model_config:
            del weak_model_config["al_strategy"]
        if "n_al_rounds" in weak_model_config:
            del weak_model_config["n_al_rounds"]
        if use_default_lr:
            weak_model_config["lr"] = MODELS_DICT[weak_model_size].default_lr

        weak_model_config_name = get_config_foldername(weak_model_config)

        weak_labels_path = (
            results_folder + "/" + sweep_subfolder + "/" + weak_model_config_name + "/weak_labels"
        )

    eval_batch_size = model_config.eval_batch_size
    random.seed(seed)

    # Load dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]

    if weak_labels_path is None:
        split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
        train1_ds, train2_ds = split_data["train"], split_data["test"]
        print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))
        config_name = get_config_foldername(config)
    else:
        if not weak_labels_path.endswith("weak_labels"):
            weak_labels_path = weak_labels_path + "/weak_labels"
        if mix_ratio is not None:  # and mix_ratio != 1.0 and mix_ratio != 0.0:
            if al_strategy is not None:
                print(
                    f"AL mix model, size {model_size}: scoring with checkpoint, "
                    f"replacing top {mix_ratio:.0%} uncertain ({al_strategy}) with GT labels, "
                    f"weak labels for remaining {1-mix_ratio:.0%}"
                )
            else:
                print(
                    f"Training mix model, size {model_size} on {mix_ratio:.0%} strong ({mix_selection}) + {1-mix_ratio:.0%} weak labels from weak model {weak_model_size}"
                )
        if sync_command is not None:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(
                ["download", weak_labels_path.replace("/weak_labels", ""), results_folder]
            )
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        train1_ds = load_from_disk(weak_labels_path)
        if al_strategy is None and mix_ratio is not None:
            train1_ds = mix_labels(train1_ds, strong_frac=mix_ratio, seed=seed, selection=mix_selection)
        train2_ds = None

        weak_model_config = json.load(open(weak_labels_path.replace("weak_labels", "config.json")))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        if al_strategy is not None and resume_from_checkpoint is None:
            w2s_base_config = {k: v for k, v in config.items()
                               if k not in ("mix_ratio", "al_strategy", "mix_selection", "n_al_rounds")}
            resume_from_checkpoint = os.path.join(
                results_folder, sweep_subfolder, get_config_foldername(w2s_base_config)
            )
            print(f"Inferred resume_from_checkpoint: {resume_from_checkpoint}")
        config["weak_model"] = weak_model_config

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)
    logger.configure(
        name="{sweep_subfolder}_{config_name}_{datetime_now}",
        save_path=save_path,
        sweep_subfolder=sweep_subfolder,
        config_name=config_name,
    )
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)

    # Tokenize shared eval datasets before branching (needed by all training paths).
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer, max_ctx)
    loss_fn = loss_dict[loss]

    if al_strategy is not None:
        # AL path: score → label top-k → train, repeated n_al_rounds times.
        # n_al_rounds=1 is the single-round case.
        assert resume_from_checkpoint is not None, "Could not infer resume_from_checkpoint; pass it explicitly"
        assert mix_ratio is not None

        train1_tok = tokenize_dataset(train1_ds, tokenizer, max_ctx)
        n_pool = len(train1_tok)
        per_round_k = max(1, round((mix_ratio / n_al_rounds) * n_pool))
        labeled_indices: set = set()
        current_checkpoint = resume_from_checkpoint

        if n_al_rounds > 1:
            print(
                f"\nMulti-round AL: {n_al_rounds} rounds × {per_round_k} labels/round "
                f"= {per_round_k * n_al_rounds}/{n_pool} ({mix_ratio:.0%} budget), strategy={al_strategy}"
            )

        for round_idx in range(n_al_rounds):
            is_last = round_idx == n_al_rounds - 1
            round_save = save_path if is_last else os.path.join(save_path, f"round_{round_idx + 1:02d}")

            if n_al_rounds > 1:
                print(f"\n{'='*50}")
                print(f"AL Round {round_idx + 1}/{n_al_rounds}  |  scoring {n_pool - len(labeled_indices)} unlabeled examples")

            unlabeled_mask = [i for i in range(n_pool) if i not in labeled_indices]
            unlabeled_tok = train1_tok.select(unlabeled_mask)
            weak_soft = np.array(unlabeled_tok["soft_label"]) if al_strategy == "disagreement" else None
            scored = _score_pool(model_config, current_checkpoint, unlabeled_tok, eval_batch_size)
            round_scores = _score_examples(
                np.array(scored["soft_label"]), al_strategy,
                weak_soft_labels=weak_soft, seed=seed + round_idx,
            )
            new_labeled = {unlabeled_mask[i] for i in np.argsort(-round_scores)[:per_round_k]}
            labeled_indices |= new_labeled
            if n_al_rounds > 1:
                print(
                    f"  +{len(new_labeled)} → {len(labeled_indices)}/{n_pool} "
                    f"({len(labeled_indices) / n_pool:.1%}) labeled"
                )

            # Reveal GT labels for all accumulated labeled examples; weak labels remain for the rest.
            def _reveal(ex, idx, _li=labeled_indices):
                if idx in _li:
                    gt = int(ex["gt_label"])
                    return {"soft_label": [1.0 - float(gt), float(gt)]}
                return {}

            mixed_tok = train1_tok.map(_reveal, with_indices=True, load_from_cache_file=False)

            if n_al_rounds > 1:
                print(f"  Training round {round_idx + 1}/{n_al_rounds}...")
                logger.configure(
                    name="{sweep_subfolder}_{config_name}_{datetime_now}",
                    save_path=round_save,
                    sweep_subfolder=sweep_subfolder,
                    config_name=config_name,
                )
            else:
                print(f"Training model model, size {model_size}")
            test_results, weak_ds = train_and_save_model(
                model_config,
                mixed_tok,
                test_ds,
                inference_ds=None,
                batch_size=batch_size,
                save_path=round_save,
                loss_fn=loss_fn,
                lr=lr,
                epochs=epochs,
                force_retrain=force_retrain,
                eval_batch_size=eval_batch_size,
                minibatch_size_per_device=minibatch_size_per_device,
                train_with_dropout=train_with_dropout,
                linear_probe=linear_probe,
                lr_schedule=lr_schedule,
                optimizer_name=optim,
                eval_every=eval_every,
                resume_from_checkpoint=current_checkpoint,
            )
            current_checkpoint = round_save
            if n_al_rounds > 1 and test_results is not None:
                round_acc = np.mean([x["acc"] for x in test_results])
                print(f"  Round {round_idx + 1}/{n_al_rounds} accuracy: {round_acc:.4f}")

    else:
        # No AL: tokenize and train directly.
        train1_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx)
        print(f"Training model model, size {model_size}")
        test_results, weak_ds = train_and_save_model(
            model_config,
            train1_ds,
            test_ds,
            inference_ds=train2_ds,
            batch_size=batch_size,
            save_path=save_path,
            loss_fn=loss_fn,
            lr=lr,
            epochs=epochs,
            force_retrain=force_retrain,
            eval_batch_size=eval_batch_size,
            minibatch_size_per_device=minibatch_size_per_device,
            train_with_dropout=train_with_dropout,
            linear_probe=linear_probe,
            lr_schedule=lr_schedule,
            optimizer_name=optim,
            eval_every=eval_every,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    if weak_ds is not None:
        weak_ds.save_to_disk(save_path + "/" + "weak_labels")


    acc = np.mean([x["acc"] for x in test_results])
    res_dict = {"accuracy": acc}
    print("accuracy:", acc)

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(save_path, "results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=2)

    if sync_command is not None:
        print("Syncing results to remote storage...")
        try:
            sync_command_list = sync_command.split(" ")
            sync_command_list.extend(["upload", save_path, results_folder])
            print(f"Running sync command: {' '.join(sync_command_list)}")
            result = subprocess.run(sync_command_list, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Sync command failed with return code {result.returncode}")
        except Exception as e:
            raise RuntimeError("Failed to sync results to remote storage.") from e

    return res_dict


if __name__ == "__main__":
    fire.Fire(main)
