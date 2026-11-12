# CUAD adapter (checkpoint-2091)

The fine-tuned weights Node 3 and Node 4 use: a LoRA adapter (rank 16) on
`unsloth/gemma-4-E2B-it`, trained on CUAD with
[`../Training Procedure/Finetuning-gemma4-e2b-for-cuad.ipynb`](../Training%20Procedure).

```
checkpoint-2091/
  adapter_config.json          LoRA settings
  adapter_model.safetensors    the weights (~101 MB)
```

Only the files needed to run the model are here (no optimizer or trainer state).
The Kaggle runner and `pipeline.PipelineConfig` load them from this folder by default,
so no Kaggle dataset is needed.

Built on Google's Gemma 4; using these weights is subject to the base model's license.
