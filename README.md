# Interaction-benchmark

## Scenario Retrieval Benchmark


**Step 1: Get backbone features**

```
cd datasets
python features.py
```

**Step 1: Run model with saved backbone features**

```
cd retrieval
python new_train.py --id model_name --seq_len 16 --scale 4 --bce 4 --weight 20 -- epochs 100 --lr 5e-4
```

