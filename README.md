## Generating Domain Knowledge
```
generating_domain_knowledge_no_DO_clean.py

```
## Train
```
python train_button.py --config data/config/train_sh_dim76_units96_h4c512.yaml
```

## Test
First of all, download the trained model and extract it to the path:`data/checkpoint`.

```
python evaluate_att.py --config data/checkpoint/eval_sh_dim76_units96_h4c512.yaml
```

This code is based on the implementation from https://github.com/HCPLab-SYSU/HIAM and https://github.com/Yigang0622/Metro-Transfer-Algorithm.
