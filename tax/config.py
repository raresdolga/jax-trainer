"""Config object passed to trainer and models.
"""

from typing import Literal, Dict, Any
from flax import struct
import yaml

LR_DECAY = Literal["cosine"] | Literal["linear"] | Literal["constant"]
EMBED_TYPE = Literal["rope"] | Literal["xpos"] | Literal["absolute"] | Literal["nope"]


@struct.dataclass
class Config:
    """
    High Level config with options for trainer.
    """

    @classmethod
    def load(cls, yaml_file, **kwargs):
        """Read configuration from json
        Args:
        yaml_file: Union[str, os.PathLike]
            Path to the yaml config
        **kwargs: other config parameters which are not part of the config file.
            If they are both part of the config and passed in the method,
                the parameters from the config file will take precendence
        """
        with open(yaml_file, "r", encoding="utf-8") as reader:
            config = yaml.safe_load(reader)
        # update json file configs with the bash configs
        config.update(kwargs)

        config = cls.validate(config)
        return cls(**config)

    @classmethod
    def validate(cls, config: Dict[str, Any]):
        """Check if attributes are correct

        Args:
            config (Comfig): Con

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if not "name" in config:
            raise NotImplementedError(
                "Experiemnt must have a name. Default not supported"
            )
        if not "base_dir" in config:
            raise NotImplementedError(
                "Experiemnt must have a base_dir. Default not supported"
            )
        return config

    # Name of the experiment.
    name: str
    # base directory where to dump trainign output. Experiment name will be a subfolder here.
    base_dir: str
    # The project under which run is saved
    project: str = "diffusion"
    # The team/account under which the project is saved
    entity: str = "dummy_project"
    # tokenizer path: used for pretrained tokenizers
    tokenizer_path: str = None
    # name of the dataset used for classification
    dataset_name: str = "shakespeare"
    # number of epochs to train the VAE for
    epochs: int = 10
    # number of train steps. Should be set to None if we want to use epochs
    train_steps: int = None
    # number of steps between evaluation
    eval_steps: int = 10
    # number of samples in validation data to use for evaluation
    eval_samples: int = None
    # The maximum number of checkpoints to save
    max_checkpoints: int = 1
    # The learning rate"
    lr: float = 3e-4
    # small lerning rate for models like ssm
    small_lr: float = None
    # percentage of steps to do warmup out of total steps
    warmup_pc: float = 0
    # exact number of warmup steps, takes precedance over warmup_pc
    # weight decay for optimizer
    weight_decay: float = 0.01
    warmup: int = 0
    # learning rate decay function
    lr_decay_fn: LR_DECAY = "cosine"
    # training precission
    mixed_precision: Literal["bf16"] | None = None
    # end value used only for linear decay learning rate
    lr_end_value: float = 0.00001
    # use batchnorm or layer norm
    batchnorm: float = False
    shuffle_train: bool = True
    # batch size per all devices
    batch_size: int = 32
    # gradient accumulation steps
    grad_accumulation_steps: int = 1
    # Path to the pretrained checkpoint, useful for resuming training
    check_path: str = None
    # wandb run id to resume if checkpath specified
    run_id: str = None
    # Whether to use wandb logging
    wandb_log: bool = False
    # whether to process data with hugging face from scratch or not.
    # True = use cached versions
    disable_cache: bool = False


@struct.dataclass
class ModelConfig(Config):
    """Configration for building a model"""

    # type of embedding
    embed_type: EMBED_TYPE = "absolute"
    # number layers
    nlayers: int = 6
    # number heads
    nheads: int = 4
    num_key_value_heads: int = 4
    # attention dimension
    head_dim: int = None
    # Hidden dimension
    hidden_dim: int = 128
    intermediate_dim: int = 2048
    # Dimention for the rotation matrix
    L: int = 10
    latte_nheads: int = 1
    # State dimension mamba models
    state_dim: int = 128
    # number of unrolls used for the scan operations in latte
    unroll: int = 100
    # maximum sequence length:
    max_seq_len: int = 1024
    # local attention for latte machiatto
    att_block_len: int = 128
    # maximu length used in positional embedings
    pos_embed_max_len: int = 1024
    # dropout each layer
    dropout_att: float = 0.0
    dropout: float = 0.1
    # used for initialisation
    initializer_range: float = 0.02
    # normalize before or after mlp
    prenorm: bool = False


@struct.dataclass
class LRAConfig(ModelConfig):
    """Properties specific to Long range Arena"""

    pool: str = "mean"
    num_classes: int = 2


@struct.dataclass
class LMConfig(ModelConfig):
    """Properties specific to Long range Arena"""

    vocab_size: int = None
    attention_bias: bool = False
