# Issue: Eval GIF only contains 4-5 frames despite a 400-step rollout

## Summary

The end-of-training eval gif feature added to `mappo_rnn_overcooked_v3_full_obs.py`
(see `run_eval_episode` + the per-seed block at the end of `single_run`) runs without
error, logs successfully to wandb, but the resulting `.gif` file only contains 4-5
frames, not the ~401 frames (1 reset state + up to 400 step states) the rollout
should be producing. There is visible motion in those few frames, but it stops far
short of a full episode. This needs to be root-caused inside `animate()` /
`grid_rendering_v2.py` / the gif-writing path, which is outside the training script
and wasn't directly inspectable during the original implementation (no local jaxmarl
checkout was available).

## Evidence collected so far

**The rollout itself produced real, varied states.** A debug print comparing the
first and last collected state in `state_seq` (added temporarily to
`regenerate_eval_gif.py`) found genuine differences, not a frozen/static episode.

**Actions look sane but are highly repetitive in early steps.** Debug prints from
inside `run_eval_episode` for the first several steps of one run showed:

```
step 0: env_act = {'agent_0': Array(5, dtype=int32), 'agent_1': Array(3, dtype=int32)}
step 0: done_dict = {'__all__': False, 'agent_0': False, 'agent_1': False}, reward = {'agent_0': 0.0, 'agent_1': 0.0}
step 1: env_act = {'agent_0': Array(5, dtype=int32), 'agent_1': Array(3, dtype=int32)}
step 1: done_dict = {'__all__': False, ...}, reward = {...: 0.0}
step 2: env_act = {'agent_0': Array(5, dtype=int32), 'agent_1': Array(3, dtype=int32)}
... (identical actions repeated through step 4)
```

Both agents pick the exact same action every step shown (`agent_0` always action
`5`, `agent_1` always action `3`). `done["__all__"]` was `False` throughout, so the
rollout was NOT terminating early — this rules out the earlier theory that
`run_eval_episode`'s `break` on `done["__all__"]` was firing prematurely. (This was
confirmed on a run against an *older* checkpoint that produced a 3-frame gif; we
don't yet have the equivalent step-by-step log from the *latest* retrained run that
produced the 4-5 frame gifs analyzed below, since that terminal output wasn't saved.)

**XLA log from the successful retrain confirms ~401 frames WERE rendered.** During
the most recent full retrain (no errors, eval gif logged successfully to wandb), the
following appeared in stderr:

```
Constant folding an instruction is taking > 1s:
  %divide.5426 = f32[401,4,5]{2,1,0} divide(...), metadata={op_name="jit(_render_state)/jit(main)/div" source_file=".../jaxmarl/viz/grid_rendering_v2.py" source_line=117}
```

The `401` in that shape is the vmap batch dimension `_render_state` was traced
over — i.e., `animate()` did receive and attempt to render a stack of 401 states
(matching a full max_steps=400 episode + 1 reset state). This was a one-time slow
compile warning, not an error, and is not itself the bug.

**But the gif files written to disk only contain 4-5 frames.** Two separate gif
outputs from two different training runs were inspected directly (frame count,
per-frame diffs, durations):

| File | Frames | Frames differing from frame 0 | Per-frame pixel diff |
|---|---|---|---|
| run A | 5 | 4 | 314, 314, 314, 598 |
| run B | 4 | 3 | 516, 516, 830 |

So: 401 states went into `animate()`, real per-frame rendering happened (confirmed
by the XLA log), and yet only 4-5 frames exist in the final `.gif` on disk. The
frame loss is happening somewhere between `_render_state`'s vmapped output and the
file `animate()` writes — i.e., inside `OvercookedV3Visualizer.animate()` itself,
not in the training script's code that calls it.

## What this rules out

- Not `run_eval_episode` breaking early on `done["__all__"]` (confirmed `False` for
  the steps logged; XLA shape of 401 confirms a near-full episode was rendered).
- Not a totally frozen/static policy (states do differ, frames do show motion,
  just very little of it makes it into the final file).
- Not an exception/crash (`wandb.log` succeeded, no traceback, gif file is valid
  and openable).

## Leading hypothesis

`animate()` (or something it calls — possibly imageio's gif writer, or a
deduplication/optimization step within `OvercookedV3Visualizer`) may be collapsing
or dropping frames that are pixel-identical or near-identical to their neighbors.
Given the early-step action repetition observed (`agent_0`/`agent_1` picking the
same action for at least 5 consecutive steps), it's plausible the trained policy is
stuck against a wall/counter for long stretches of the 400-step episode, producing
long runs of visually-identical consecutive frames — if `animate()` or imageio
de-duplicates these, that could explain ~401 input frames collapsing to single
digits of *visually distinct* output frames. This is a hypothesis, not yet
confirmed against `animate()`'s actual source.

Alternative/secondary hypotheses, roughly in order of plausibility:
- `animate()` accepts a `state_seq` arg but only renders a Python-level subsample
  of it (e.g. every Nth frame, or a fixed small max-frame count) regardless of
  vmap batch size — possible if the vmapped render shape (401) and the number of
  frames actually fed to the gif writer are determined by two different,
  decoupled pieces of logic inside `animate()`.
- The gif writer (`imageio`) is being called with a `duration`/`fps` parameter
  that's misinterpreted, causing most frames to be written with 0 duration and
  then collapsed/optimized away by something downstream (a GIF viewer or PIL
  re-save), though the durations inspected directly from the file
  (`[None, 0, 0, 190, 0]`) suggest the frames themselves are simply not present in
  the file, not present-but-zero-duration.

## Proposed plan for the in-repo agent

1. **Read `OvercookedV3Visualizer.animate()`'s actual source** in
   `jaxmarl/viz/overcooked_v3_visualizer.py` (this was not available when the
   training-script-side code was written; only a method signature list and grep
   results were available, not the implementation). Specifically check:
   - Does it subsample/limit frames internally (look for any slicing, `[::N]`,
     `max_frames`, or similar before the imageio write call)?
   - Does it deduplicate consecutive identical frames before writing?
   - What exact imageio call is used to write the gif, and with what `duration`/
     `fps` arguments?

2. **Add a direct frame-count assertion/print inside `animate()`** (temporarily,
   for debugging) immediately before the imageio write call — print
   `len(frames)` or `frames.shape[0]` right there, to compare against the 401
   that `_render_state` was vmapped over. This will confirm whether the drop
   happens inside `animate()`'s Python logic (frames computed but not all
   written) or inside imageio/the gif codec itself.

3. **If `animate()` does no deduplication and frame count going into imageio's
   write call already matches 401**, then the leading hypothesis is wrong and the
   issue is in the imageio call itself (gif codec settings, duration handling,
   or an imageio version difference) — at that point, try writing frames as an
   `.mp4` instead via the same `state_seq`/render path (if `animate()` or a
   sibling method supports it) to isolate whether this is gif-codec-specific.

4. **Separately, and regardless of the above:** investigate *why* the trained
   policy appears to pick the same action repeatedly for several consecutive
   steps early in the eval episode. This may be entirely correct behavior (e.g.
   agent walking toward a fixed target, or genuinely stuck against a counter)
   but is worth a sanity check against the actual `cramped_room` layout and
   reward curve from training, independent of the frame-count bug. If the policy
   is meaningfully under-trained or stuck, that's a separate, real issue from the
   gif-encoding bug above and shouldn't be conflated with it.

## Relevant code (training-script side, already implemented and not believed to
be the source of this particular bug, included for reference)

```python
def run_eval_episode(actor_params, actor_network, env, rng, config, max_steps):
    num_actors = env.num_agents
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng)
    hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
    done = jnp.zeros((num_actors,), dtype=bool)
    state_seq = [env_state]

    for step in range(max_steps):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            -1, *env.observation_space().shape
        )
        ac_in = (obs_batch[None, :], done[None, :])
        hstate, pi = actor_network.apply(actor_params, hstate, ac_in)

        if config.get("USE_ACTION_MASK", False):
            env_action_mask = env.action_mask(env_state)
            action_mask = env_action_mask.reshape((1, num_actors, env.num_actions))
            pi = distrax.Categorical(logits=jnp.where(action_mask, pi.logits, -1e9))

        action = jnp.argmax(pi.logits, axis=-1)
        env_act = unbatchify(action, env.agents, 1, env.num_agents)
        env_act = {k: v.reshape(()) for k, v in env_act.items()}

        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done_dict, info = env.step(_rng, env_state, env_act)
        state_seq.append(env_state)
        done = jnp.full((num_actors,), done_dict["__all__"])
        if bool(done_dict["__all__"]):
            break
    return state_seq
```

```python
# inside single_run, per seed:
gif_path = os.path.join(save_dir, f"final_episode_seed{i}.gif")
stacked_state_seq = jax.tree_util.tree_map(
    lambda *leaves: jnp.stack(leaves), *state_seq
)
viz.animate(stacked_state_seq, filename=gif_path)
wandb.log({f"eval/final_episode_seed{i}": wandb.Video(gif_path, format="gif")})
```

`stacked_state_seq` converts the Python list returned by `run_eval_episode` into a
single pytree with a leading axis of length `len(state_seq)` (confirmed to be ~401
via the XLA shape log), since `animate()`'s internal `_render_state` call is
`jax.vmap`'d and needs a stacked pytree, not a list of separate pytrees.
