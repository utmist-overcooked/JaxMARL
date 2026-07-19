# IC3Net for JaxMARL

JAX/Flax implementation of IC3Net and related communication baselines (IC, IRIC, CommNet).

## Quick Start

```bash
# Train IC3Net on MPE Simple Spread
python baselines/IC3Net/ic3net_train.py

# Train on Overcooked
python baselines/IC3Net/ic3net_train.py --config-name=ic3net_overcooked_medium_test

# Run inference
python baselines/IC3Net/ic3net_infer.py --config-name=ic3net_overcooked_medium_infer

# Visualize
python baselines/IC3Net/visualize_overcooked.py
```

## Models

All models support both **feedforward** and **recurrent (LSTM)** variants:

- **IC / IRIC**: Independent controllers, no communication (IRIC adds individual rewards)
- **CommNet**: Continuous communication
- **IC3Net**: CommNet with hard-attention gating (talk/silent)

Default: LSTM recurrent models (`RECURRENT: true`).

## Running the different models

The model is selected by the `BASELINE` key in the Hydra config, so switching
models is just a matter of pointing `--config-name` at a different YAML from
`config/` (the naming convention is `<baseline>_<env>_<difficulty>.yaml`):

```bash
# IC3Net (default config is ic3net_mpe.yaml)
python baselines/IC3Net/ic3net_train.py

# CommNet on the same task
python baselines/IC3Net/ic3net_train.py --config-name=commnet_mpe

# IC / IRIC (no communication)
python baselines/IC3Net/ic3net_train.py --config-name=ic_mpe
python baselines/IC3Net/ic3net_train.py --config-name=iric_mpe
```

Alternatively, override individual keys on the command line without a new
config file:

```bash
python baselines/IC3Net/ic3net_train.py BASELINE=commnet RECURRENT=false
```

Inference works the same way — use the matching `*_infer.yaml` config (or
override `BASELINE`/`MODEL_PATH` directly) and make sure `BASELINE` and
`RECURRENT` match the settings the checkpoint was trained with, since they
determine the network architecture that the weights are loaded into.

## Configuration

Key parameters in `config/*.yaml`:

- `BASELINE`: `"ic"`, `"iric"`, `"commnet"`, or `"ic3net"`
- `RECURRENT`: `true` for LSTM, `false` for feedforward
- `HIDDEN_DIM`: Hidden layer size
- `COMM_PASSES`: Communication rounds

## Reference

```
Singh, A., Jain, T., & Sukhbaatar, S. (2018). 
Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks.
arXiv:1812.09755
```

- Uses RMSprop optimizer as per original paper
- REINFORCE with value baseline (not PPO)
- Communication adds minimal overhead due to JAX optimizations

## Citation

If you use this implementation, please cite:

```bibtex
@misc{singh2018learningcommunicatescalemultiagent,
      title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks}, 
      author={Amanpreet Singh and Tushar Jain and Sainbayar Sukhbaatar},
      year={2018},
      eprint={1812.09755},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
}
```
