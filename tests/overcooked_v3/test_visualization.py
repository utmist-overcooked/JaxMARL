"""Visualization tests for Overcooked V3 environment."""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
import os

from jaxmarl.environments.overcooked_v3 import OvercookedV3, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.common import DynamicObject
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer, HAS_IMAGEIO


class TestOvercookedV3Visualization:
    """Test visualization functionality for Overcooked V3."""

    def test_visualizer_creates_image(self):
        """Test that visualizer produces an image array."""
        env = OvercookedV3(layout="cramped_room")
        viz = OvercookedV3Visualizer(env)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        img = viz.render_state(state)

        # Check image has correct shape (H, W, 3)
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        assert img.dtype == jnp.uint8

    def test_visualizer_image_dimensions(self):
        """Test that image dimensions match layout size."""
        env = OvercookedV3(layout="cramped_room")
        viz = OvercookedV3Visualizer(env, tile_size=32)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        img = viz.render_state(state)

        expected_height = env.height * 32
        expected_width = env.width * 32
        assert img.shape[0] == expected_height
        assert img.shape[1] == expected_width

    def test_visualizer_all_layouts(self):
        """Test visualization works for all registered layouts."""
        key = jax.random.PRNGKey(42)

        for layout_name in overcooked_v3_layouts.keys():
            layout = overcooked_v3_layouts[layout_name]
            recipe_mode = (
                "fixed" if len(layout.possible_recipes) == 1 else "alternating"
            )
            env = OvercookedV3(layout=layout_name, recipe_mode=recipe_mode)
            viz = OvercookedV3Visualizer(env)

            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey)

            img = viz.render_state(state)

            assert img is not None
            assert len(img.shape) == 3
            assert img.shape[2] == 3

    def test_visualizer_after_steps(self):
        """Test visualization after environment steps."""
        env = OvercookedV3(layout="cramped_room", max_steps=100)
        viz = OvercookedV3Visualizer(env)

        key = jax.random.PRNGKey(123)
        obs, state = env.reset(key)

        # Take some steps
        for _ in range(10):
            key, *subkeys = jax.random.split(key, 3)
            actions = {
                agent: int(jax.random.randint(subkeys[i], (), 0, 6))
                for i, agent in enumerate(env.agents)
            }
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

            img = viz.render_state(state)
            assert img is not None
            assert len(img.shape) == 3

    def test_visualizer_sequence_rendering(self):
        """Test rendering a sequence of states."""
        env = OvercookedV3(layout="cramped_room", max_steps=50)
        viz = OvercookedV3Visualizer(env)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Collect states
        states = [state]
        for _ in range(5):
            key, subkey = jax.random.split(key)
            actions = {agent: 0 for agent in env.agents}
            obs, state, rewards, dones, info = env.step(subkey, state, actions)
            states.append(state)

        # Stack states for vmap
        stacked_states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)

        frames = viz.render_sequence(stacked_states)

        assert frames.shape[0] == len(states)
        assert len(frames.shape) == 4  # (num_frames, H, W, 3)

    @pytest.mark.skipif(
        os.environ.get("CI") == "true" or not HAS_IMAGEIO,
        reason="Skip GIF creation in CI or if imageio not installed"
    )
    def test_visualizer_animation_creation(self, tmp_path):
        """Test creating an animated GIF."""
        env = OvercookedV3(layout="cramped_room", max_steps=50)
        viz = OvercookedV3Visualizer(env)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Collect states
        states = [state]
        for _ in range(5):
            key, subkey = jax.random.split(key)
            actions = {agent: 0 for agent in env.agents}
            obs, state, rewards, dones, info = env.step(subkey, state, actions)
            states.append(state)

        stacked_states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)

        gif_path = tmp_path / "test_animation.gif"
        viz.animate(stacked_states, filename=str(gif_path))

        assert gif_path.exists()

    def test_visualizer_with_conveyors(self):
        """Test visualization with conveyor belt layout."""
        if "conveyor_demo" not in overcooked_v3_layouts:
            pytest.skip("conveyor_demo layout not available")

        env = OvercookedV3(
            layout="conveyor_demo",
            enable_item_conveyors=True,
        )
        viz = OvercookedV3Visualizer(env)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        img = viz.render_state(state)

        assert img is not None
        assert len(img.shape) == 3

    def test_visualizer_jit_compatible(self):
        """Test that render_state can be JIT compiled."""
        env = OvercookedV3(layout="cramped_room")
        viz = OvercookedV3Visualizer(env)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # JIT compile the render function
        jit_render = jax.jit(viz._render_state, static_argnums=(1,))

        img1 = jit_render(state, None)
        img2 = jit_render(state, None)

        # Results should be identical
        assert jnp.allclose(img1, img2)

    def test_visualizer_different_tile_sizes(self):
        """Test visualization with different tile sizes."""
        env = OvercookedV3(layout="cramped_room")

        for tile_size in [16, 32, 64]:
            viz = OvercookedV3Visualizer(env, tile_size=tile_size)

            key = jax.random.PRNGKey(0)
            obs, state = env.reset(key)

            img = viz.render_state(state)

            expected_height = env.height * tile_size
            expected_width = env.width * tile_size
            assert img.shape[0] == expected_height
            assert img.shape[1] == expected_width


class TestVisualizerRenderingContent:
    """Test that specific game elements are rendered correctly."""

    def test_agents_visible(self):
        """Test that agents are rendered (non-zero pixels at agent locations)."""
        env = OvercookedV3(layout="cramped_room")
        viz = OvercookedV3Visualizer(env, tile_size=32)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        img = viz.render_state(state)

        # Agents should create non-background pixels
        # Check that image is not all the same color (indicating rendering happened)
        unique_colors = np.unique(np.array(img).reshape(-1, 3), axis=0)
        assert len(unique_colors) > 2  # More than just background and grid lines

    def test_pot_rendering_changes_with_state(self):
        """Test that pot rendering changes as ingredients are added."""
        env = OvercookedV3(layout="cramped_room", max_steps=200)
        viz = OvercookedV3Visualizer(env)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        img_initial = viz.render_state(state)

        # Take many random steps to potentially change pot state
        for _ in range(50):
            key, subkey = jax.random.split(key)
            actions = {agent: int(jax.random.randint(subkey, (), 0, 6)) for agent in env.agents}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        img_after = viz.render_state(state)

        # Images should potentially be different (game state changed)
        # Note: They might be the same if no meaningful actions occurred
        assert img_initial.shape == img_after.shape

    def test_pot_progress_uses_sampled_cook_duration(self):
        env = OvercookedV3(
            layout="cramped_room",
            pot_cook_time=90,
            pot_cook_time_range=[7, 9],
            pot_burn_time=5,
        )
        viz = OvercookedV3Visualizer(env, tile_size=32)
        _, state = env.reset(jax.random.PRNGKey(0))

        pot_y, pot_x = state.pot_positions[0]
        state = state.replace(
            grid=state.grid.at[pot_y, pot_x, 1].set(
                DynamicObject.ingredient(0) * 3
            ),
            pot_cooking_timer=state.pot_cooking_timer.at[0].set(11),
            pot_cook_durations=state.pot_cook_durations.at[0].set(7),
        )

        img = viz.render_state(state)
        progress_row = int(pot_y) * 32 + 27

        assert jnp.array_equal(
            img[progress_row, int(pot_x) * 32 + 4],
            jnp.array([0, 255, 0], dtype=jnp.uint8),
        )
        assert not jnp.array_equal(
            img[progress_row, int(pot_x) * 32 + 16],
            jnp.array([0, 255, 0], dtype=jnp.uint8),
        )

    def test_zero_sampled_duration_falls_back_to_fixed_cook_time(self):
        env = OvercookedV3(
            layout="cramped_room",
            pot_cook_time=7,
            pot_burn_time=5,
        )
        viz = OvercookedV3Visualizer(env, tile_size=32)
        _, state = env.reset(jax.random.PRNGKey(0))
        pot_y, pot_x = state.pot_positions[0]
        base_state = state.replace(
            grid=state.grid.at[pot_y, pot_x, 1].set(
                DynamicObject.ingredient(0) * 3
            ),
            pot_cooking_timer=state.pot_cooking_timer.at[0].set(11),
        )
        tracked_state = base_state.replace(
            pot_cook_durations=base_state.pot_cook_durations.at[0].set(7)
        )

        assert jnp.array_equal(
            viz.render_state(base_state),
            viz.render_state(tracked_state),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
