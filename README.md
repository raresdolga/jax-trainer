# jax-trainer
An efficient distributed FSDP in jax

# Running Examples
## Single host trainer
Run imdb movie classification from the long range arena dataset with batch parallelisation given by jit function:
```
pdm run bash ./tax/examples/bin/run_lra.sh $PWD imdb.yaml lra_imdb
```

## FSDP Trainer
Train a gemma language model with the FSDP trainer
```
pdm run bash ./tax/examples/bin/run_lm.sh $PWD lm.yaml test_trainer
```