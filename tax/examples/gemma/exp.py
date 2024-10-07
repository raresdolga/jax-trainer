"""
Data Parallel Model sharding implementation of lm
"""

from functools import partial
import os
import wandb
import dataclasses
import json
import jax
from jax import numpy as jnp
import numpy as np
from datasets import disable_caching

from transformers import AutoTokenizer
from tax.examples.gemma.lm_dp import TinyStories
from tax.train.dfsp_jax import (
    DFSDPTrainer,
    shard_module_params,
)
from tax.examples.gemma.task import Gemma
from tax.evals.lang_eval import LanguageEvaluator
from tax.examples.utils import parse_args
from tax.config import LMConfig

# minimum size at which sharding parameters starts
MIN_WEIGHT_SHARD_SIZE = 2**10


def get_dp(config: LMConfig) -> TinyStories:
    """_summary_

    Args:
        config (LMConfig): model and trainer configuration

    Raises:
        ValueError: Not implemenyed dataset

    Returns:
        TinyStories: DataProcessor
    """
    if config.dataset_name == "tiny-stories":
        cache_dir = os.path.join(config.base_dir, "input", "tiny-sories")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-2b",
            cache_dir=os.path.join(config.base_dir, "input/cache_hugg"),
            truncation_side="right",
            padding_side="right",
        )
        dp = TinyStories(
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            max_seq_len=config.max_seq_len,
            num_load_procs=min(1, os.cpu_count() - 1),
        )
        raw_data = dp.get_raw_data()
        tok_data = dp.tokenize(raw_data, force_preproc=config.disable_cache)

    else:
        raise ValueError("Corpus not supported!")
    return dp, tokenizer, raw_data, tok_data


class LMTask:
    """Language modeling with Gemma architecture"""

    def __init__(self, config) -> None:
        print(f"Config is {config}")
        self.config = config
        self.report_to = "none"
        self.wandb_run = None

        self.out_dir = os.path.join(
            self.config.base_dir, "out_models", self.config.name
        )
        os.makedirs(self.out_dir, exist_ok=True)
        # dump config file in model dir for debug
        with open(
            os.path.join(self.out_dir, "config.json"), "w+", encoding="utf-8"
        ) as f:
            a = dataclasses.asdict(config)
            json.dump(a, f)
        self.set_logger()

    def set_logger(self):
        """Set wandb logger"""
        # configure wandb logs
        if self.config.wandb_log:
            resume = False
            run_id = None
            if not self.config.check_path is None:
                resume = "must"
                run_id = self.config.run_id
            wandb_run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.name,
                dir=self.out_dir,
                config=self.config,
                id=run_id,
                resume=resume,
            )
            self.report_to = "wandb"
            self.wandb_run = wandb_run

    def update_config(self, tokenizer):
        self.config = self.config.replace(vocab_size=tokenizer.vocab_size)

    def get_model(self, tokenizer: AutoTokenizer, sharded: bool = True):
        """Created and shard model

        Args:
            tokenizer (AutoTokenizer): tokenizer used to tokenize the text
            sharded (bool, optional): Shard model or not. Defaults to True.

        Returns:
            _type_: _description_
        """
        match self.config.mixed_precision:
            case "bf16":
                mixed_precission = jnp.bfloat16
            case _:
                mixed_precission = jnp.float32

        constructor = Gemma

        if sharded:
            sharded_model = shard_module_params(
                constructor, axis_name="B", min_weight_size=MIN_WEIGHT_SHARD_SIZE
            )
        else:
            sharded_model = constructor

        model = sharded_model(
            self.config,
            pad_id=tokenizer.pad_token_id,
            dtype=mixed_precission,
            sharded=sharded,
        )
        print(model)
        return model

    def train(self, train_rng: jax.random.PRNGKey):
        """Train Model

        Args:
            train_rng (jax.random.PRNGKey): trainiong random key
        """
        dp, tokenizer, _, tokenized_data = get_dp(self.config)
        data_collator = dp.get_collate_fn(
            return_type="np", max_seq_len=self.config.max_seq_len
        )
        # add additional info like vocab size
        self.update_config(tokenizer)

        model = self.get_model(tokenizer, sharded=True)
        train_rng, init_rng, eval_rng = jax.random.split(train_rng, 3)
        evaluator = LanguageEvaluator(
            model,
            tokenizer,
            tokenized_data["validation"],
            data_collator=data_collator,
            config=self.config,
            rng=eval_rng,
        )
        trainer = DFSDPTrainer(
            config=self.config,
            out_dir=self.out_dir,
            model=model,
            train_data=tokenized_data["train"],
            data_collator=data_collator,
            evaluator=evaluator,
            wandb_run=self.wandb_run,
            rng=init_rng,
            model_inputs_orded=("input_ids", "labels"),
        )
        if not self.config.check_path is None:
            trainer.train(train_rng, checkpoint_path=self.config.check_path)
        else:
            trainer.train(train_rng)


def main():
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, train_rng = jax.random.split(rng)
    args = parse_args()
    config = LMConfig.load(
        yaml_file=args.config_file, base_dir=args.base_dir, name=args.name
    )

    if config.disable_cache:
        print("Disabling Cache")
        disable_caching()

    task = LMTask(config)
    if args.evaluate:
        raise Exception("Not yet Implemented")
    else:
        task.train(train_rng)


if __name__ == "__main__":
    main()
