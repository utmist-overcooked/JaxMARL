# Plan: End-of-Run GIF Generation for `mappo_rnn_overcooked_v3_full_obs.py`
 
## Goal
 
After a training run completes, run one deterministic evaluation episode with the
final trained policy, capture every step as a rendered frame, and log all frames to
the same wandb run under one key. This is the *only* GIF mechanism — no periodic
gifs during training, no per-checkpoint gifs. The existing
`Generate_gifs_from_logged_images_on_wandb.ipynb` notebook then consumes that run
unmodified to produce the actual `.gif`/`.mp4` files.
 
This plan does not include final code, because one upstream dependency
(`OvercookedV3Visualizer`'s exact render API) has not yet been confirmed against the
real environment in a runnable way. Once that's settled, the implementation is
mechanical — the spec below should be enough for that final pass.
 
---
 
## What's already true, and what isn't
 
The uploaded `mappo_rnn_overcooked_v3_full_obs.py` (1045 lines) currently has:
 
- No `OvercookedV3Visualizer` import.
- No render/viz/imageio/gif code anywhere.
- A per-update `callback` (called via `jax.debug.callback`) that logs scalar metrics
  with `wandb.log(metric)` and periodically saves `.safetensors` checkpoints every
  `checkpoint_interval` updates.
- A vectorized training loop (`NUM_ENVS` environments stepping in lockstep inside
  `jax.lax.scan`) with no concept of "one clean episode" — there is nothing to
  retroactively extract frames from after the fact.
A grep against the user's actual `~/JaxMARL` repo surfaced a *different, longer*
version of this same script (`baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py`,
with `OvercookedV3Visualizer` usage around line 853) that does not match the
uploaded file byte-for-byte. The uploaded file is confirmed, by diff and re-check,
to be the shorter/older variant with no visualizer code at all. **This means the
plan below assumes a fresh implementation, not an adaptation of existing viz code.**
If the longer repo version is what should actually be modified, that file needs to
be uploaded directly (not the standalone copy) before final code is written.
 
---
 
## Confirmed facts about the visualizer
 
From `grep -rn "class.*Visualizer" jaxmarl/` and a method listing of
`jaxmarl/viz/overcooked_v3_visualizer.py`:
 
```
class OvercookedV3Visualizer:
    def __init__(self, env, tile_size=TILE_PIXELS, subdivs=3):
    def _lazy_init_window(self):
    def show(self, block=False):
    def render(self, state, agent_view_size=None):
    def render_state(self, state, agent_view_size=None):
    def animate(self, state_seq, filename="animation.gif", agent_view_size=None):
    def render_sequence(self, state_seq, agent_view_size=None):
    ...
```
 
The file imports `from jaxmarl.viz.window import Window` and
`import jaxmarl.viz.grid_rendering_v2 as rendering`, and optionally `imageio`
(`HAS_IMAGEIO` flag, gracefully degrades if not installed).
 
Existing call sites across the repo (demo scripts, `tests/overcooked_v3/test_visualization.py`,
`jaxmarl/environments/overcooked_v3/interactive.py`) all construct it as
`OvercookedV3Visualizer(env)` or `OvercookedV3Visualizer(env, tile_size=tile_size)`.
 
**There is also an `animate(state_seq, filename="animation.gif", agent_view_size=None)`
method that writes a gif directly from a sequence of states.** This is highly
relevant: it may make the entire "render every step → log as wandb.Image list →
reassemble in the notebook" pipeline unnecessary. It's possible the simplest
correct implementation is:
 
1. Collect the `state` at every step of the eval episode into a list (`state_seq`).
2. Call `viz.animate(state_seq, filename="final_episode.gif")` once, locally.
3. Upload that single `.gif` file directly via `wandb.log({"eval/final_episode_gif": wandb.Video("final_episode.gif")})`
   (or `wandb.save(...)` for a raw file artifact).
This would **replace** the per-frame `wandb.Image` list approach and make the
existing gif-notebook irrelevant for this particular use case, since the gif would
already exist as a finished artifact, not something to be reconstructed from many
uploaded PNGs. This needs to be decided before final implementation — see Open
Questions below.
 
---
 
## Open questions (blocking final implementation)
 
1. **Use `animate()` directly, or render frames manually and reconstruct via the
   notebook?**
   - `animate()` is less code, already handles gif assembly, and was clearly built
     for exactly this purpose.
   - Manual per-frame `wandb.Image` logging matches the originally-discussed
     pipeline (compatible with the existing notebook) but duplicates work
     `animate()` already does, and adds ~400 file uploads to wandb instead of one
     `.gif` file.
   - **Recommendation: use `animate()` directly** unless there's a reason to want
     the raw per-frame PNGs on wandb too (e.g. for building custom dashboards later).
2. **Return type and side effects of `render()` / `render_state()` / `animate()`
   — unconfirmed.**
   - Does `render(state)` return a numpy array, a PIL Image, or something else?
   - Does instantiating `OvercookedV3Visualizer(env)` require a display backend
     (the `Window` import suggests pygame/similar)? If training runs on a headless
     remote machine (no `$DISPLAY`), this constructor call may fail or hang unless
     there's an offscreen/headless rendering mode.
   - Does `animate()` need `_lazy_init_window()` to have been called first, or does
     it work standalone?
   - **Action needed:** run a tiny standalone test on the actual training machine
     before wiring this into the training script:
```python
     from jaxmarl.environments.overcooked_v3 import OvercookedV3
     from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
     import jax
 
     env = OvercookedV3(layout="cramped_room")
     viz = OvercookedV3Visualizer(env)
     key = jax.random.PRNGKey(0)
     obs, state = env.reset(key)
     viz.animate([state, state, state], filename="test.gif")
```
     If this fails with a display/window error on the headless training machine,
     a workaround (e.g. `SDL_VIDEODRIVER=dummy`, or checking if `Window` is lazily
     skipped when `show()` is never called) needs to be found first.
 
3. **Greedy action extraction from `distrax.Categorical`.**
   - The actor outputs `pi = distrax.Categorical(logits=...)`. For a deterministic
     eval rollout, need `pi.mode()` (if available on this distrax version) or
     `jnp.argmax(pi.logits, axis=-1)` as a fallback. Action-masking logic
     (`USE_ACTION_MASK`) from the training loop should also be replicated in eval
     if it's enabled in config, otherwise the eval policy may select masked-out
     actions that behave differently from training.
4. **Multi-seed runs (`NUM_SEEDS > 1`).**
   - Current test config has `NUM_SEEDS: 1`, so this doesn't matter yet, but the
     final code path needs an explicit decision: one gif for seed 0 only, or one
     gif per seed (`NUM_SEEDS` separate `animate()` calls / wandb log entries)?
   - **Recommendation:** loop over all seeds and log one gif per seed with a
     seed-indexed key (e.g. `eval/final_episode_seed{i}`), since the existing
     checkpoint-saving loop already iterates `for i, rng in enumerate(rngs)` and
     this would be a natural place to slot in.
5. **Episode length / early termination.**
   - `ENV_KWARGS.max_steps` is 400 in the test config. Episodes can end before
     that via `done["__all__"]`. The eval rollout should stop collecting states
     once `done["__all__"]` is true, rather than always capturing exactly 400
     frames (which would either pad with repeated terminal states or run into an
     auto-reset depending on how the raw `OvercookedV3.step` behaves outside
     `LogWrapper`).
6. **Which env instance to use for eval — the wrapped or unwrapped env?**
   - Training uses `env = LogWrapper(OvercookedV3(**config["ENV_KWARGS"]))`. The
     visualizer is constructed as `OvercookedV3Visualizer(env)` in all repo
     examples, always with a raw `OvercookedV3` instance, not a wrapped one.
   - **Recommendation:** construct a second, unwrapped `OvercookedV3(**config["ENV_KWARGS"])`
     specifically for eval+visualization, separate from the wrapped/vmapped/jitted
     training env. Cheap to instantiate, avoids passing `LogWrapper`-specific state
     into a visualizer that expects raw `OvercookedV3` state.
---
 
## Implementation plan (once open questions are resolved)
 
### 1. Imports
Add to the top of `mappo_rnn_overcooked_v3_full_obs.py`:
```python
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
```
 
### 2. New function: `run_eval_episode`
Standalone function, not inside `make_train` (runs outside JAX's
jit/scan/vmap machinery — a plain Python loop, since it executes once per training
run, not once per step):
 
```python
def run_eval_episode(actor_params, actor_network, env, rng, config, max_steps):
    """
    Roll out one deterministic episode with the given actor params.
    Returns a list of env states (for OvercookedV3Visualizer.animate)
    or a list of rendered RGB frames, depending on the approach chosen
    in Open Question 1.
    """
    # reset
    # initialize hidden state via ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    # loop up to max_steps:
    #   build (obs, done) actor input
    #   hidden, pi = actor_network.apply(actor_params, hidden, ac_in)
    #   action = greedy action from pi (see Open Question 3)
    #   step env
    #   append state (or rendered frame) to list
    #   break early if done["__all__"]
    # return collected list
```
 
### 3. Call site: end of `single_run`, after the existing checkpoint-saving loop
Right after:
```python
save_params(actor_params, actor_path)
save_params(critic_params, critic_path)
print(f"Saved actor params to {actor_path}")
print(f"Saved critic params to {critic_path}")
```
add (per-seed, inside the existing `for i, rng in enumerate(rngs):` loop):
```python
if config["WANDB_MODE"] != "disabled":
    eval_env = OvercookedV3(**config["ENV_KWARGS"])  # unwrapped, fresh instance
    viz = OvercookedV3Visualizer(eval_env)
    eval_rng = jax.random.PRNGKey(config["SEED"] + 1000 + i)
    state_seq = run_eval_episode(
        actor_params, actor_network, eval_env, eval_rng, config,
        max_steps=config["ENV_KWARGS"]["max_steps"],
    )
    gif_path = os.path.join(save_dir, f"final_episode_seed{i}.gif")
    viz.animate(state_seq, filename=gif_path)
    wandb.log({f"eval/final_episode_seed{i}": wandb.Video(gif_path, format="gif")})
```
(Exact wandb logging call — `wandb.Video` vs `wandb.Image` list vs `wandb.save` —
depends on the Open Question 1 decision.)
 
### 4. No changes needed to:
- The per-update `callback` function.
- `checkpoint_interval` logic.
- The yaml config (no new keys required — this is unconditional on
  `WANDB_MODE != "disabled"`, matching the existing pattern elsewhere in the script).
- The gif-generation notebook, **if** going the `animate()` + `wandb.Video` route
  (the notebook becomes unnecessary for this specific feature, since the gif is
  already a finished artifact). If going the per-frame `wandb.Image` route instead,
  the notebook works unmodified as originally planned.
---
 
## Other considerations
 
- **Headless rendering risk.** If training runs on a remote/cloud machine without a
  display, `OvercookedV3Visualizer`'s `Window` dependency could fail. This must be
  tested standalone (see Open Question 2) before relying on it inside a long
  training job — discovering a display error after a multi-hour run finishes would
  be a frustrating failure mode.
- **Action masking parity.** If `config.get("USE_ACTION_MASK", False)` is true
  during training, the eval rollout should apply the same masking logic
  (`env.action_mask`) when computing greedy actions, or the eval policy's behavior
  won't match what was actually trained.
- **Cost/timing.** Capturing every step of a 400-step episode and assembling a gif
  happens once, at the very end of `single_run`, after the (typically much longer)
  training loop. This adds a small, bounded amount of wall-clock time at the end of
  the run — not spread out during training — and should be negligible relative to
  total run time.
- **File output location.** The gif file written by `animate()` should go in the
  same `save_dir` (`{WANDB_DIR}/models`) as the `.safetensors` checkpoints and the
  saved config yaml, for consistency with existing output organization.
- **Imageio dependency.** `HAS_IMAGEIO` is checked at import time in the
  visualizer module; if `imageio` isn't installed in the training environment,
  `animate()` may silently no-op or raise — worth confirming `imageio` is in the
  project's dependencies before relying on this path.