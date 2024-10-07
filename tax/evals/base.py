import abc
from typing import Iterable, Callable, Any, Dict, Tuple
import jax
from torch.utils.data import DataLoader
from tqdm import tqdm

PyTree = Any


class Evaluator(abc.ABC):
    """Class for evaluating a model"""

    def __init__(
        self, val_data: Iterable, data_collator: Callable, config: PyTree
    ) -> None:
        super().__init__()
        self._config = config

        self._val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=True,
        )

    @abc.abstractmethod
    def compute_metrics(self, *args, **kwargs) -> Dict[str, jax.Array]:
        """Calculate metrics given the output of the model.

        Returns:
            Dict[str, jax.Array]: Output metrics
        """

    def evaluate(
        self,
        trainer_eval_fn: Callable[[str, jax.Array], Tuple[jax.Array]],
        prefix="eval_",
        **kwargs,
    ) -> Dict[str, jax.Array]:
        """Iterate over validation data, get outputs from trainer eval and compute metrics.
            Decouple from trainer to add data-specific evaluation logic:
                - squad split in overlapping windows
                - language do generation from promts
        Args:
            trainer_eval_fn: Callable[Dict[str, np.array]] -> Tuple
                Function which places data to devices by trainer sharding.
                Contains platform specific model call. Outputs "labels", "model_output"
            prefix: str - used to rename metrics depending on eval/test data
        Returns:
            Dict[str, jax.Array]: Output metrics
        """
        scores = {}
        progress_bar = tqdm(
            range(len(self._val_loader)), position=0, leave=True, initial=0
        )
        for batch in self._val_loader:
            labels, output = trainer_eval_fn(batch)
            metrics = self.compute_metrics(output, labels)
            for k in metrics.keys():
                scores[k] = scores.get(k, 0) + metrics[k]
            progress_bar.update(1)

        scores = {prefix + k: v / len(self._val_loader) for k, v in scores.items()}

        return scores
