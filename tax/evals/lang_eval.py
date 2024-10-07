"""Evaluators for language generation"""

from typing import Callable, Tuple, Dict
import jaxtyping as jt
from tqdm import tqdm
import math
from jax import numpy as jnp
import jax
from flax import linen as nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tax.config import ModelConfig
from .base import Evaluator


def pred_acc_lm(output: Dict[str, jt.Array]) -> Dict[str, jt.Array]:
    """Compute merics language modelling

    Args:
        output (Dict[str, jt.Array]): Output of the model

    Returns:
        Dict[str, jt.Array]: Metrics
    """
    loss = output["loss"]
    metrics = {
        "loss": loss,
        "bpc": loss / math.log(2),
        "ppl": jnp.exp(loss),
    }
    return metrics


class LanguageEvaluator(Evaluator):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        val_data: Dataset,
        data_collator: Callable,
        config: ModelConfig,
        rng,
    ) -> None:
        super().__init__(val_data, data_collator, config)

        self._tokenizer = tokenizer
        self._model = model
        self._batchnorm = config.batchnorm
        self._batch_size = config.batch_size
        self._config = config
        self._rng = rng
        self._total_samples = (
            len(self._val_loader)
            if self._config.eval_samples is None
            else self._config.eval_samples
        )
        assert self._total_samples <= len(self._val_loader)

    def evaluate(
        self,
        trainer_eval_fn: Callable[[str, jax.Array], Tuple[jax.Array]],
        prefix="eval_",
        **kwargs,
    ) -> Dict[str, jax.Array]:
        """Iterate over validation data, get outputs from trainer eval
        and compute metrics.
        Decouple from trainer to add data-specific evaluation logic:
            - squad split in overlapping windows
            - language do generation from promts
        Args:
            trainer_eval_fn (Callable[[Dict[str, jt.Array]], Tuple])
                Function which places data to devices by trainer sharding.
                Contains platform specific model call. Outputs "labels", "model_output"
            prefix (str): used to rename metrics depending on eval/test data
            state (TrainerState):  used for post eval
        """
        scores = {}
        print(self._config.eval_samples)
        progress_bar = tqdm(
            range(self._total_samples), position=0, leave=True, initial=0
        )
        it = 0
        val_iter = iter(self._val_loader)
        while it < self._total_samples:
            it += 1
            batch = next(val_iter)
            labels, output = trainer_eval_fn(batch)
            metrics = self.compute_metrics(output, labels)
            for k in metrics.keys():
                scores[k] = scores.get(k, 0) + metrics[k]
            progress_bar.update(1)

        scores = {prefix + k: v / self._total_samples for k, v in scores.items()}
        # print(
        #     "Final mean: ", np.mean(all_tines[3:]), "Final Var: ", np.std(all_tines[3:])
        # )
        return scores

    def compute_metrics(self, output, labels):
        return pred_acc_lm(output)
