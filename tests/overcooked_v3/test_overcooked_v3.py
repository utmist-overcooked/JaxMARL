"""Tests for Overcooked V3 environment."""

import jax
import jax.numpy as jnp
import pytest
from jaxmarl import make
from jaxmarl.environments.overcooked_v3 import OvercookedV3, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.common import (
    DynamicObject, StaticObject, Direction, Position, Agent, SoupType, Actions,
)
from jaxmarl.environments.overcooked_v3.settings import DELIVERY_REWARD
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


class TestOvercookedV3API:
    """Test that OvercookedV3 implements the MultiAgentEnv interface."""

    def test_inherits_base_env(self):
        env = OvercookedV3()
        assert isinstance(env, MultiAgentEnv)

    def test_has_required_attributes(self):
        env = OvercookedV3()
        assert hasattr(env, 'agents')
        assert hasattr(env, 'num_agents')
        assert hasattr(env, 'action_spaces')
        assert hasattr(env, 'observation_spaces')

    def test_reset_returns_correct_format(self):
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Check obs is dict with all agent keys
        assert isinstance(obs, dict)
        for agent in env.agents:
            assert agent in obs

        # Check state has required fields
        assert hasattr(state, 'time')
        assert hasattr(state, 'terminal')
        assert hasattr(state, 'grid')
        assert hasattr(state, 'agents')

    def test_step_returns_correct_format(self):
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Random actions
        actions = {agent: 0 for agent in env.agents}
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(subkey, state, actions)

        # Check dones has __all__ key
        assert "__all__" in dones

        # Check all agents have entries
        for agent in env.agents:
            assert agent in obs
            assert agent in rewards
            assert agent in dones


class TestOvercookedV3Rollout:
    """Test that random rollouts work correctly."""

    def test_random_rollout_completes(self):
        """Run a full episode with random actions."""
        env = OvercookedV3(max_steps=100)
        key = jax.random.PRNGKey(42)

        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)

        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < 200:
            # Random actions for all agents
            key, *subkeys = jax.random.split(key, len(env.agents) + 1)
            actions = {
                agent: int(jax.random.randint(subkeys[i], (), 0, 6))
                for i, agent in enumerate(env.agents)
            }

            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

            total_reward += sum(rewards.values())
            done = dones["__all__"]
            step_count += 1

        assert step_count > 0
        print(f"Rollout completed in {step_count} steps, total reward: {total_reward}")

    def test_jit_compiled_rollout(self):
        """Verify rollout works with JIT compilation."""
        env = OvercookedV3(max_steps=50)

        @jax.jit
        def step_fn(key, state, actions):
            return env.step(key, state, actions)

        @jax.jit
        def reset_fn(key):
            return env.reset(key)

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        obs, state = reset_fn(subkey)

        for _ in range(10):
            actions = {agent: 0 for agent in env.agents}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = step_fn(subkey, state, actions)

        assert True  # If we get here without error, JIT works

    def test_vmap_parallel_envs(self):
        """Verify environment works with vmap for parallel rollouts."""
        env = OvercookedV3()
        num_envs = 4

        keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
        reset_fn = jax.vmap(env.reset)
        obs, states = reset_fn(keys)

        # Check batched dimensions
        assert obs[env.agents[0]].shape[0] == num_envs


class TestOvercookedV3Layouts:
    """Test layout loading and parsing."""

    def test_cramped_room_layout(self):
        env = OvercookedV3(layout="cramped_room")
        assert env.num_agents == 2
        assert env.height > 0
        assert env.width > 0

    def test_all_registered_layouts(self):
        """Test that all registered layouts can be loaded."""
        for layout_name in overcooked_v3_layouts.keys():
            env = OvercookedV3(layout=layout_name)
            key = jax.random.PRNGKey(0)
            obs, state = env.reset(key)
            assert obs is not None
            assert state is not None


class TestOvercookedV3PotMechanics:
    """Test pot cooking and burning mechanics."""

    def test_pot_initial_state(self):
        """Verify pots start empty."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # All pot timers should be 0
        assert jnp.all(state.pot_cooking_timer == 0)

    def test_pot_cooking_timer_decrements(self):
        """Verify pot cooking timer decrements when pot is full."""
        env = OvercookedV3(pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Manually set up a pot with 3 ingredients and timer started
        # Find pot position
        pot_y, pot_x = state.pot_positions[0]

        # Add full ingredients to pot
        from jaxmarl.environments.overcooked_v3.common import DynamicObject
        full_pot = DynamicObject.ingredient(0) * 3  # 3 onions
        new_grid = state.grid.at[pot_y, pot_x, 1].set(full_pot)
        new_timers = state.pot_cooking_timer.at[0].set(10)
        state = state.replace(grid=new_grid, pot_cooking_timer=new_timers)

        # Take a step (no-op actions)
        actions = {agent: 4 for agent in env.agents}  # stay action
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        # Timer should have decremented
        assert new_state.pot_cooking_timer[0] == 9


class TestOvercookedV3OrderQueue:
    """Test order queue system."""

    def test_order_queue_disabled_by_default(self):
        """Verify order queue is disabled by default."""
        env = OvercookedV3()
        assert env.enable_order_queue == False

    def test_order_queue_can_be_enabled(self):
        """Verify order queue can be enabled."""
        env = OvercookedV3(enable_order_queue=True)
        assert env.enable_order_queue == True
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        assert state.order_types is not None


class TestOvercookedV3Conveyors:
    """Test conveyor belt mechanics."""

    def test_item_conveyors_disabled_by_default(self):
        """Verify item conveyors are disabled by default."""
        env = OvercookedV3()
        assert env.enable_item_conveyors == False

    def test_player_conveyors_disabled_by_default(self):
        """Verify player conveyors are disabled by default."""
        env = OvercookedV3()
        assert env.enable_player_conveyors == False

    def test_conveyor_demo_layout(self):
        """Test conveyor demo layout loads correctly."""
        env = OvercookedV3(layout="conveyor_demo", enable_item_conveyors=True)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Should have some item conveyors
        assert jnp.any(state.item_conveyor_active_mask)


class TestOvercookedV3Registration:
    """Test environment registration."""

    def test_make_function_works(self):
        """Verify env can be created via jaxmarl.make()."""
        env = make("overcooked_v3")
        assert env is not None
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')

    def test_make_with_kwargs(self):
        """Verify make works with custom kwargs."""
        env = make("overcooked_v3", max_steps=200)
        assert env.max_steps == 200


class TestOvercookedV3Observations:
    """Test observation generation."""

    def test_observation_shape(self):
        """Verify observation has correct shape."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        for agent in env.agents:
            assert obs[agent].shape == env.obs_shape

    def test_partial_observability(self):
        """Test partial observability with agent_view_size."""
        env = OvercookedV3(agent_view_size=2)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Observation should be smaller than full grid
        assert obs[env.agents[0]].shape[0] <= 5
        assert obs[env.agents[0]].shape[1] <= 5


class TestOvercookedV3Actions:
    """Test action handling."""

    def test_movement_actions(self):
        """Test that movement actions change agent position."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        initial_pos = (state.agents.pos.x[0].item(), state.agents.pos.y[0].item())

        # Try moving right
        actions = {env.agents[0]: 0, env.agents[1]: 4}  # right for agent 0, stay for agent 1
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        # Position might have changed (depends on layout)
        new_pos = (new_state.agents.pos.x[0].item(), new_state.agents.pos.y[0].item())
        # Direction should have changed to RIGHT regardless of movement
        assert new_state.agents.dir[0] == 2  # Direction.RIGHT

    def test_stay_action(self):
        """Test that stay action doesn't change position."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        initial_pos = (state.agents.pos.x[0].item(), state.agents.pos.y[0].item())
        initial_dir = state.agents.dir[0].item()

        # Stay action for all agents
        actions = {agent: 4 for agent in env.agents}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        new_pos = (new_state.agents.pos.x[0].item(), new_state.agents.pos.y[0].item())

        assert initial_pos == new_pos
        assert initial_dir == new_state.agents.dir[0].item()


def _find_goal_position(state):
    """Find the (y, x) position of the first GOAL tile."""
    goal_mask = state.grid[:, :, 0] == StaticObject.GOAL
    ys, xs = jnp.where(goal_mask, size=1)
    return int(ys[0]), int(xs[0])


def _place_agent_adjacent_to_goal(state, agent_idx, goal_y, goal_x):
    """Place agent one cell above the goal, facing down toward it."""
    adj_y, adj_x = goal_y - 1, goal_x
    new_pos = Position(x=jnp.array([adj_x]), y=jnp.array([adj_y]))
    new_dir = state.agents.dir.at[agent_idx].set(Direction.DOWN)
    new_pos_x = state.agents.pos.x.at[agent_idx].set(new_pos.x[0])
    new_pos_y = state.agents.pos.y.at[agent_idx].set(new_pos.y[0])
    new_agents = state.agents.replace(
        pos=Position(x=new_pos_x, y=new_pos_y),
        dir=new_dir,
    )
    return state.replace(agents=new_agents)


def _set_agent_inventory(state, agent_idx, item):
    """Set an agent's inventory to a specific item."""
    new_inv = state.agents.inventory.at[agent_idx].set(item)
    return state.replace(agents=state.agents.replace(inventory=new_inv))


def _inject_order(state, slot, soup_type, expiration):
    """Inject an order into a specific slot."""
    return state.replace(
        order_types=state.order_types.at[slot].set(soup_type),
        order_expirations=state.order_expirations.at[slot].set(expiration),
        order_active_mask=state.order_active_mask.at[slot].set(True),
    )


class TestOrderQueueDelivery:
    """Test order queue integration with delivery system."""

    def _make_env(self, **kwargs):
        defaults = dict(
            enable_order_queue=True,
            max_orders=5,
            order_generation_rate=0.0,  # Disable random generation for deterministic tests
            order_expiration_time=200,
            max_steps=500,
        )
        defaults.update(kwargs)
        return OvercookedV3(**defaults)

    def _setup_delivery(self, env, state, agent_idx, soup_type):
        """Set up agent adjacent to goal with a plated soup matching soup_type."""
        goal_y, goal_x = _find_goal_position(state)
        state = _place_agent_adjacent_to_goal(state, agent_idx, goal_y, goal_x)
        plated_recipe = SoupType.to_plated_recipe(soup_type)
        state = _set_agent_inventory(state, agent_idx, plated_recipe)
        return state

    def test_delivery_matches_single_order(self):
        """Delivering soup that matches a queued order gives reward and deactivates order."""
        env = self._make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Inject one onion soup order
        state = _inject_order(state, slot=0, soup_type=int(SoupType.ONION_SOUP), expiration=100)

        # Set up agent 0 to deliver onion soup
        state = self._setup_delivery(env, state, agent_idx=0, soup_type=int(SoupType.ONION_SOUP))

        # Agent 0 interacts (deliver), agent 1 stays
        actions = {env.agents[0]: int(Actions.interact), env.agents[1]: int(Actions.stay)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        # Each agent gets the shared delivery reward
        agent_reward = float(rewards[env.agents[0]])
        assert agent_reward == pytest.approx(DELIVERY_REWARD, abs=1.0)

        # Order should be deactivated
        assert not new_state.order_active_mask[0]
        assert new_state.order_types[0] == int(SoupType.NONE)

    def test_delivery_no_matching_order_no_reward(self):
        """Delivering soup with no matching order gives no reward."""
        env = self._make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Inject tomato soup order
        state = _inject_order(state, slot=0, soup_type=int(SoupType.TOMATO_SOUP), expiration=100)

        # Set up agent 0 to deliver onion soup (wrong type)
        state = self._setup_delivery(env, state, agent_idx=0, soup_type=int(SoupType.ONION_SOUP))

        actions = {env.agents[0]: int(Actions.interact), env.agents[1]: int(Actions.stay)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        # No delivery reward (order penalty may apply from expiration tick but delivery itself = 0)
        agent_reward = float(rewards[env.agents[0]])
        assert agent_reward <= 0.0

        # Tomato order should still be active
        assert new_state.order_active_mask[0]

    def test_delivery_no_active_orders_no_reward(self):
        """Delivering soup with empty order queue gives no reward."""
        env = self._make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # No orders injected (all inactive by default)
        state = self._setup_delivery(env, state, agent_idx=0, soup_type=int(SoupType.ONION_SOUP))

        actions = {env.agents[0]: int(Actions.interact), env.agents[1]: int(Actions.stay)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        agent_reward = float(rewards[env.agents[0]])
        assert agent_reward == pytest.approx(0.0, abs=0.01)

    def test_earliest_expiring_order_fulfilled(self):
        """When multiple orders match, the earliest-expiring one is fulfilled."""
        env = self._make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Two onion soup orders: slot 0 expires in 100, slot 1 expires in 50
        state = _inject_order(state, slot=0, soup_type=int(SoupType.ONION_SOUP), expiration=100)
        state = _inject_order(state, slot=1, soup_type=int(SoupType.ONION_SOUP), expiration=50)

        state = self._setup_delivery(env, state, agent_idx=0, soup_type=int(SoupType.ONION_SOUP))

        actions = {env.agents[0]: int(Actions.interact), env.agents[1]: int(Actions.stay)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        # Slot 1 (expiration=50, earliest) should be deactivated
        assert not new_state.order_active_mask[1]
        # Slot 0 (expiration=100) should still be active
        assert new_state.order_active_mask[0]

    def test_order_queue_in_observations(self):
        """Order queue info should appear in observations when enabled."""
        env = self._make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        num_ingredients = env.layout.num_ingredients
        expected_extra = env.max_orders * (2 + num_ingredients)
        base_layers = 26 + 5 * num_ingredients

        # Obs shape should include order layers
        assert env.obs_shape[2] == base_layers + expected_extra

        for agent in env.agents:
            assert obs[agent].shape == env.obs_shape

    def test_obs_shape_unchanged_when_disabled(self):
        """Obs shape should not change when order queue is disabled."""
        env = OvercookedV3(enable_order_queue=False)
        num_ingredients = env.layout.num_ingredients
        expected_layers = 26 + 5 * num_ingredients
        assert env.obs_shape[2] == expected_layers

    def test_jit_compatibility(self):
        """Environment with order queue should work under JIT."""
        env = self._make_env()

        @jax.jit
        def step_fn(key, state, actions):
            return env.step(key, state, actions)

        @jax.jit
        def reset_fn(key):
            return env.reset(key)

        key = jax.random.PRNGKey(0)
        obs, state = reset_fn(key)

        state = _inject_order(state, slot=0, soup_type=int(SoupType.ONION_SOUP), expiration=100)

        for _ in range(10):
            actions = {agent: int(Actions.stay) for agent in env.agents}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = step_fn(subkey, state, actions)

        # Should complete without error
        assert True

    def test_backward_compat_disabled_order_queue(self):
        """With order queue disabled, delivery should use global recipe as before."""
        env = OvercookedV3(enable_order_queue=False, max_steps=500)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        goal_y, goal_x = _find_goal_position(state)
        state = _place_agent_adjacent_to_goal(state, 0, goal_y, goal_x)

        # Set inventory to match the global recipe (plated)
        plated_recipe = state.recipe | int(DynamicObject.PLATE) | int(DynamicObject.COOKED)
        state = _set_agent_inventory(state, 0, plated_recipe)

        actions = {env.agents[0]: int(Actions.interact), env.agents[1]: int(Actions.stay)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        agent_reward = float(rewards[env.agents[0]])
        assert agent_reward == pytest.approx(DELIVERY_REWARD, abs=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
