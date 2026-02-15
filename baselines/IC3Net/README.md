# IC3Net for JaxMARL

JAX/Flax implementation of IC3Net and related communication baselines (IC, CommNet).

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

- **IC**: Independent controllers, no communication
- **CommNet**: Continuous communication
- **IC3Net**: CommNet with hard-attention gating (talk/silent)

Default: LSTM recurrent models (`RECURRENT: true`).

## Configuration

Key parameters in `config/*.yaml`:

- `BASELINE`: `"ic"`, `"commnet"`, or `"ic3net"`
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
