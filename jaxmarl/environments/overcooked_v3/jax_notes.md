# Zac's Notes on JAX

## Overview of the process

### Single transformations

In `overcooked.py`, we define the functions on how to manipulate a **single** state (i.e. perform transformations and modifications to a state tensor). Later on, when we are training, our aim is to parallelize over thousands of environments so that the GPU can run all these calculations in parallel.

### Parallel transforamtions

See `baselines/IPPO/ippo_ff_overcooked.py`.

We first create a single env (line 134)

```python
env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
```

Then we initialize two random keys (same seed), and then make NUM_ENVS keys (line 180, 181)

```python
rng, _rng = jax.random.split(rng)                              # split off one key
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])          # make NUM_ENVS keys
```

Finally, we use `jax.vmap` to run the same function in parallel (line 182)

```python
obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
```

Similarly, we use `jax.vmap` to run `env.step` on all envs at the same time, returning a large tensor containing all the observations, rewards etc.

```python
obsv, env_state, reward, done, info = jax.vmap(
    env.step, in_axes=(0, 0, 0)                             # all 3 args are batched
)(rng_step, env_state, env_act)                             # step all in parallel
```

## Edge cases

In `overcooked.py` we use a for loop without any JIT compilation / Jax operations.

```python
for i in range(MAX_POTS):
    y, x = state.pot_positions[i]
    timer = state.pot_cooking_timer[i]
    is_active = state.pot_active_mask[i]
    pot_timer_layer = jax.lax.select(
        is_active,
        pot_timer_layer.at[y, x].set(timer),
        pot_timer_layer
    )
```

We would think this is bad but MAX_POTS is a constant so Python unrolls this into

```python
pot_timer_layer = lax.select(active[0], pot_timer_layer.at[y0, x0].set(timer0),pot_timer_layer)
pot_timer_layer = lax.select(active[1], pot_timer_layer.at[y1, x1].set(timer1),
pot_timer_layer)
pot_timer_layer = lax.select(active[2], pot_timer_layer.at[y2, x2].set(timer2),
pot_timer_layer)
pot_timer_layer = lax.select(active[3], pot_timer_layer.at[y3, x3].set(timer3),
pot_timer_layer)
```
