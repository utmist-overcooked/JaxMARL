For training macro action agents:

```
python baselines/MAPPO/mappo_macro_every_step.py \
  ENV_KWARGS.layout=pressure_gated_circuit
```

For evaluating macro agents (generating gifs + reward histograms)

```
1. Extract the actor weights:
python -m scripts.extract_actor_checkpoints --variant every_step --run-dir models/mappo_macro/mappo_macro_every_step/seed_0


2. Run the eval script
python scripts/visualize_macro_mappo_rollout.py --run-dir="models/mappo_macro/mappo_macro_every_step/seed_0" --output=outputs/without_comm/rollout.gif --checkpoint-label=checkpoint_00002500.npz --num-episodes=1 --variant=every_step

```

For training the communication prototcol:

```
python baselines/MAPPO/mappo_macro_every_step_comm.py \
  ENV_KWARGS.layout=pressure_gated_circuit
```

For evaluating macro action agents with communication (generating gifs)

```
python scripts/visualize_macro_mappo_rollout_comm.py --run-dir="models/mappo_macro/mappo_macro_every_step_comm/seed_0" --output=outputs/with_comm/rollout.gif --checkpoint-label=checkpoint_00002500.npz --num-episodes=1 

```