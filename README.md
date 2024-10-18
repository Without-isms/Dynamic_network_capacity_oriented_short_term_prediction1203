## Train
- SZMOD
```
python train.py --config data/config/train_sh_dim76_units96_h4c512.yaml
```

## Test
First of all, download the trained model and extract it to the path:`data/checkpoint`.

- SHMOD
```
python evaluate.py --config data/checkpoint/eval_sh_dim76_units96_h4c512.yaml
```
- HZMOD
```
python evaluate.py --config data/checkpoint/eval_hz_dim26_units96_h4c512.yaml
```
