"""Public Overcooked V3 environment wrapper."""

from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from jaxmarl.environments.overcooked_v3.agent_step import run_agent_action_phase
from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    Agent,
    ButtonAction,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.initialization import (
    randomize_agent_positions,
    randomize_state,
    sample_recipe,
)
from jaxmarl.environments.overcooked_v3.interactions import (
    process_interact,
    sample_pot_cook_time,
)
from jaxmarl.environments.overcooked_v3.layouts import Layout, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.movement import is_agent_walkable
from jaxmarl.environments.overcooked_v3.observations import (
    calculate_observation_shape,
    get_obs,
    get_obs_default,
    get_obs_for_type,
)
from jaxmarl.environments.overcooked_v3.reset import reset_overcooked_v3
from jaxmarl.environments.overcooked_v3.settings import (
    DEFAULT_BARRIER_DURATION,
    DEFAULT_MAX_ORDERS,
    DEFAULT_ORDER_EXPIRATION_TIME,
    DEFAULT_ORDER_GENERATION_RATE,
    DELIVERY_REWARD,
    EVENT_NAMES,
    MAX_BARRIERS,
    MAX_BUTTONS,
    MAX_BUTTON_TARGETS,
    MAX_ITEM_CONVEYORS,
    MAX_MOVING_WALLS,
    MAX_PLAYER_CONVEYORS,
    MAX_POTS,
    MAX_PRESSURE_PLATES,
    POT_BURN_TIME,
    POT_COOK_TIME,
    POT_COOK_TIME_RANGE,
)
from jaxmarl.environments.overcooked_v3.state import ObservationType, State
from jaxmarl.environments.overcooked_v3.step import is_terminal, step_overcooked_v3
from jaxmarl.environments.overcooked_v3.systems.barriers import (
    barriers_occupied,
    update_barrier_timers,
    update_pressure_plates,
)
from jaxmarl.environments.overcooked_v3.systems.conveyors import (
    move_items_on_item_conveyors,
    push_players_on_player_conveyors,
)
from jaxmarl.environments.overcooked_v3.systems.moving_walls import move_moving_walls
from jaxmarl.environments.overcooked_v3.systems.orders import process_order_queue
from jaxmarl.environments.overcooked_v3.systems.pots import update_pot_timers
from jaxmarl.environments.overcooked_v3.utils import compute_enclosed_spaces

class OvercookedV3(MultiAgentEnv):
    """Overcooked V3 environment backed by explicit functional JAX logic."""

    EVENT_NAMES = EVENT_NAMES

    def __init__(
        self,
        layout: Union[str, Layout] = "cramped_room",
        max_steps: int = 400,
        observation_type: Union[
            ObservationType, List[ObservationType]
        ] = ObservationType.DEFAULT,
        agent_view_size: Optional[int] = None,
        # Pot settings
        pot_cook_time: int = POT_COOK_TIME,
        pot_cook_time_range: Optional[Sequence[int]] = None,
        pot_burn_time: int = POT_BURN_TIME,
        # Recipe generation settings
        recipe_mode: str = "fixed",
        recipe_probs: Optional[Union[List[float], np.ndarray, chex.Array]] = None,
        # Order queue settings
        enable_order_queue: bool = False,
        max_orders: int = DEFAULT_MAX_ORDERS,
        order_generation_rate: float = DEFAULT_ORDER_GENERATION_RATE,
        order_expiration_time: int = DEFAULT_ORDER_EXPIRATION_TIME,
        # Conveyor belt settings
        enable_item_conveyors: Optional[bool] = None,
        enable_player_conveyors: Optional[bool] = None,
        # Moving wall, pressure plate, and button settings
        enable_moving_walls: Optional[bool] = None,
        enable_buttons: Optional[bool] = None,
        enable_pressure_plates: Optional[bool] = None,
        # Barrier settings
        barrier_duration: Union[int, List[int]] = DEFAULT_BARRIER_DURATION,
        # Reward settings
        delivery_reward: float = DELIVERY_REWARD,
        shaped_rewards: bool = True,
        # Random initialization
        random_reset: bool = False,
        random_agent_positions: bool = False,
    ):
        """Initialize the Overcooked V3 environment.

        Args:
            layout: Layout name or Layout object
            max_steps: Maximum steps per episode
            observation_type: Type of observation (default or featurized)
            agent_view_size: Partial observability window size (None for full)
            pot_cook_time: Steps until a full pot becomes ready (default 90)
            pot_cook_time_range: Optional inclusive [min, max] ready-time range.
                Omit or pass an empty sequence to use pot_cook_time.
            pot_burn_time: Steps in burning window before pot burns (default 60)
            recipe_mode: Recipe generation strategy: "fixed", "random", or
                "alternating". Alternating cycles through possible_recipes in
                their declared order.
            recipe_probs: Required probability distribution over
                layout.possible_recipes in random mode; forbidden otherwise.
            enable_order_queue: Whether to use order queue system
            max_orders: Maximum orders in queue
            order_generation_rate: Probability of new order each step
            order_expiration_time: Steps before order expires
            enable_item_conveyors: Whether item conveyors move items. If None,
                inferred from whether the layout contains item conveyors.
            enable_player_conveyors: Whether player conveyors push agents. If
                None, inferred from whether the layout contains player conveyors.
            enable_moving_walls: Whether moving walls move each step. If None,
                inferred from whether the layout contains moving walls.
            enable_buttons: Whether buttons can be interacted with. If None,
                inferred from whether the layout contains buttons.
            enable_pressure_plates: Whether pressure plates can be stepped on.
            barrier_duration: Duration (steps) for timed barrier deactivation.
                Can be int (same for all) or list of ints per barrier.
            delivery_reward: Reward for correct delivery
            shaped_rewards: Whether to use shaped intermediate rewards
            random_reset: Randomize state on reset
            random_agent_positions: Randomize agent positions only
        """
        if isinstance(layout, str):
            if layout not in overcooked_v3_layouts:
                raise ValueError(
                    f"Invalid layout: {layout}, "
                    f"allowed layouts: {list(overcooked_v3_layouts.keys())}"
                )
            layout = overcooked_v3_layouts[layout]
        elif not isinstance(layout, Layout):
            raise ValueError("Invalid layout, must be a Layout object or a string key")

        is_playable, validation_messages = layout.validate_playable()
        if not is_playable:
            formatted_messages = "\n".join(
                f"- {message}" for message in validation_messages
            )
            raise ValueError(f"Invalid OvercookedV3 layout:\n{formatted_messages}")

        num_agents = len(layout.agent_positions)
        super().__init__(num_agents=num_agents)

        self.height = layout.height
        self.width = layout.width
        self.layout = layout

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.action_set = jnp.array(list(Actions))

        if isinstance(observation_type, list):
            if len(observation_type) != num_agents:
                raise ValueError(
                    "Number of observation types must match number of agents"
                )
        self.observation_type = observation_type
        self.agent_view_size = agent_view_size

        self.max_steps = max_steps

        # Pot settings
        self.pot_cook_time = pot_cook_time
        cook_time_range = (
            POT_COOK_TIME_RANGE
            if pot_cook_time_range is None
            else tuple(pot_cook_time_range)
        )
        if len(cook_time_range) not in (0, 2):
            raise ValueError(
                "pot_cook_time_range must be empty or contain exactly [min, max]"
            )
        if len(cook_time_range) == 2 and min(cook_time_range) < 1:
            raise ValueError("pot_cook_time_range values must be at least 1")
        if len(cook_time_range) == 2 and cook_time_range[0] > cook_time_range[1]:
            raise ValueError("pot_cook_time_range min must be <= max")
        self.pot_cook_time_range = jnp.array(cook_time_range, dtype=jnp.int32)
        self.pot_burn_time = pot_burn_time

        # Recipe and order queue settings
        if recipe_mode not in ("fixed", "random", "alternating"):
            raise ValueError(
                "recipe_mode must be 'fixed', 'random', or 'alternating'"
            )
        self.recipe_mode = recipe_mode

        if not isinstance(enable_order_queue, (bool, np.bool_)):
            raise ValueError("enable_order_queue must be a boolean")
        self.enable_order_queue = enable_order_queue

        if isinstance(max_orders, (bool, np.bool_)) or not isinstance(
            max_orders, Integral
        ):
            raise ValueError("max_orders must be an integer")
        if max_orders < 1:
            raise ValueError("max_orders must be at least 1")
        self.max_orders = max_orders

        if isinstance(order_generation_rate, (bool, np.bool_)) or not isinstance(
            order_generation_rate, Real
        ):
            raise ValueError("order_generation_rate must be numeric")
        if not np.isfinite(order_generation_rate) or not (
            0.0 <= order_generation_rate <= 1.0
        ):
            raise ValueError("order_generation_rate must be between 0.0 and 1.0")
        self.order_generation_rate = order_generation_rate

        if isinstance(order_expiration_time, (bool, np.bool_)) or not isinstance(
            order_expiration_time, Integral
        ):
            raise ValueError("order_expiration_time must be an integer")
        if order_expiration_time < 0:
            raise ValueError("order_expiration_time must be at least 0")
        self.order_expiration_time = order_expiration_time

        # Conveyor settings
        layout_has_item_conveyors = len(layout.item_conveyor_info) > 0
        if enable_item_conveyors is None:
            self.enable_item_conveyors = layout_has_item_conveyors
        else:
            self.enable_item_conveyors = enable_item_conveyors
            if layout_has_item_conveyors and not enable_item_conveyors:
                warnings.warn(
                    "Layout contains item conveyors, but "
                    "enable_item_conveyors=False. Item conveyors will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        layout_has_player_conveyors = len(layout.player_conveyor_info) > 0
        if enable_player_conveyors is None:
            self.enable_player_conveyors = layout_has_player_conveyors
        else:
            self.enable_player_conveyors = enable_player_conveyors
            if layout_has_player_conveyors and not enable_player_conveyors:
                warnings.warn(
                    "Layout contains player conveyors, but "
                    "enable_player_conveyors=False. Player conveyors will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        # Moving wall and button settings
        layout_has_moving_walls = len(layout.moving_wall_info) > 0
        if enable_moving_walls is None:
            self.enable_moving_walls = layout_has_moving_walls
        else:
            self.enable_moving_walls = enable_moving_walls
            if layout_has_moving_walls and not enable_moving_walls:
                warnings.warn(
                    "Layout contains moving walls, but "
                    "enable_moving_walls=False. Moving walls will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        layout_has_buttons = len(layout.button_info) > 0
        if enable_buttons is None:
            self.enable_buttons = layout_has_buttons
        else:
            self.enable_buttons = enable_buttons
            if layout_has_buttons and not enable_buttons:
                warnings.warn(
                    "Layout contains buttons, but enable_buttons=False. "
                    "Buttons will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        layout_has_pressure_plates = len(layout.pressure_plate_info) > 0
        if enable_pressure_plates is None:
            self.enable_pressure_plates = layout_has_pressure_plates
        else:
            self.enable_pressure_plates = enable_pressure_plates
            if layout_has_pressure_plates and not enable_pressure_plates:
                warnings.warn(
                    "Layout contains pressure plates, but "
                    "enable_pressure_plates=False. Pressure plates will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        # Barrier settings
        self.barrier_duration = barrier_duration

        # Reward settings
        self.delivery_reward = delivery_reward
        self.shaped_rewards_enabled = shaped_rewards

        # Random reset
        self.random_reset = random_reset
        self.random_agent_positions = random_agent_positions

        # Pre-compute possible recipes
        self.possible_recipes = jnp.array(layout.possible_recipes, dtype=jnp.int32)
        self._validate_recipe_configuration(recipe_mode, recipe_probs)
        self.recipe_probs = (
            None
            if recipe_probs is None
            else jnp.array(recipe_probs, dtype=jnp.float32)
        )
        self._order_recipe_encodings = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=jnp.int32),
                jax.vmap(DynamicObject.get_recipe_encoding)(self.possible_recipes),
            )
        )

        # Pre-compute enclosed spaces for random agent placement
        self.enclosed_spaces = compute_enclosed_spaces(
            layout.static_objects == StaticObject.EMPTY,
        )

        # Compute observation shape
        self.obs_shape = calculate_observation_shape(
            self.width,
            self.height,
            self.layout,
            self.observation_type,
            self.agent_view_size,
            self.enable_order_queue,
            self.max_orders,
        )

        # Extract pot positions from layout
        pot_mask = layout.static_objects == StaticObject.POT
        pot_indices = np.argwhere(pot_mask)
        self.num_pots = min(len(pot_indices), MAX_POTS)
        self._pot_positions = np.zeros((MAX_POTS, 2), dtype=np.int32)
        self._pot_active_mask = np.zeros(MAX_POTS, dtype=bool)
        for i, (y, x) in enumerate(pot_indices[:MAX_POTS]):
            self._pot_positions[i] = [y, x]
            self._pot_active_mask[i] = True

        # Goal positions are fixed by layout and used for post-plating distance
        # shaping. Keep them as a static array so JIT-compiled steps can compute
        # nearest-goal Euclidean distance without scanning the grid dynamically.
        goal_indices = np.argwhere(layout.static_objects == StaticObject.GOAL)
        self._goal_positions = goal_indices.astype(np.int32)

        # Extract conveyor info from layout
        self._item_conveyor_positions = np.zeros(
            (MAX_ITEM_CONVEYORS, 2), dtype=np.int32
        )
        self._item_conveyor_directions = np.zeros(MAX_ITEM_CONVEYORS, dtype=np.int32)
        self._item_conveyor_active_mask = np.zeros(MAX_ITEM_CONVEYORS, dtype=bool)
        for i, (y, x, direction) in enumerate(
            layout.item_conveyor_info[:MAX_ITEM_CONVEYORS]
        ):
            self._item_conveyor_positions[i] = [y, x]
            self._item_conveyor_directions[i] = direction
            self._item_conveyor_active_mask[i] = True

        self._player_conveyor_positions = np.zeros(
            (MAX_PLAYER_CONVEYORS, 2), dtype=np.int32
        )
        self._player_conveyor_directions = np.zeros(
            MAX_PLAYER_CONVEYORS, dtype=np.int32
        )
        self._player_conveyor_active_mask = np.zeros(MAX_PLAYER_CONVEYORS, dtype=bool)
        for i, (y, x, direction) in enumerate(
            layout.player_conveyor_info[:MAX_PLAYER_CONVEYORS]
        ):
            self._player_conveyor_positions[i] = [y, x]
            self._player_conveyor_directions[i] = direction
            self._player_conveyor_active_mask[i] = True

        # Extract moving wall info from layout
        self._moving_wall_positions = np.zeros((MAX_MOVING_WALLS, 2), dtype=np.int32)
        self._moving_wall_directions = np.zeros(MAX_MOVING_WALLS, dtype=np.int32)
        self._moving_wall_active_mask = np.zeros(MAX_MOVING_WALLS, dtype=bool)
        self._moving_wall_bounce = np.zeros(MAX_MOVING_WALLS, dtype=bool)
        for i, (y, x, direction, bounce) in enumerate(
            layout.moving_wall_info[:MAX_MOVING_WALLS]
        ):
            self._moving_wall_positions[i] = [y, x]
            self._moving_wall_directions[i] = direction
            self._moving_wall_active_mask[i] = True
            self._moving_wall_bounce[i] = bounce

        # Extract button info from layout
        self._button_positions = np.zeros((MAX_BUTTONS, 2), dtype=np.int32)
        self._button_target_idxs = np.zeros(
            (MAX_BUTTONS, MAX_BUTTON_TARGETS), dtype=np.int32
        )
        self._button_target_mask = np.zeros(
            (MAX_BUTTONS, MAX_BUTTON_TARGETS), dtype=bool
        )
        self._button_action_type = np.zeros(MAX_BUTTONS, dtype=np.int32)
        self._button_active_mask = np.zeros(MAX_BUTTONS, dtype=bool)
        for i, (y, x, target_idxs, action_type) in enumerate(
            layout.button_info[:MAX_BUTTONS]
        ):
            self._button_positions[i] = [y, x]
            if isinstance(target_idxs, list):
                target_idxs = tuple(target_idxs)
            elif not isinstance(target_idxs, tuple):
                target_idxs = (target_idxs,)
            for j, target_idx in enumerate(target_idxs[:MAX_BUTTON_TARGETS]):
                self._button_target_idxs[i, j] = target_idx
                self._button_target_mask[i, j] = True
            self._button_action_type[i] = action_type
            self._button_active_mask[i] = True

        self._moving_wall_initial_paused = np.zeros(MAX_MOVING_WALLS, dtype=bool)
        for button_idx in range(MAX_BUTTONS):
            if (
                self._button_active_mask[button_idx]
                and self._button_action_type[button_idx] == ButtonAction.TRIGGER_MOVE
            ):
                for target_slot in range(MAX_BUTTON_TARGETS):
                    if self._button_target_mask[button_idx, target_slot]:
                        target_idx = self._button_target_idxs[button_idx, target_slot]
                        self._moving_wall_initial_paused[target_idx] = True

        # Extract barrier info from layout
        self._barrier_positions = np.zeros((MAX_BARRIERS, 2), dtype=np.int32)
        self._barrier_initial_active = np.zeros(MAX_BARRIERS, dtype=bool)
        self._barrier_active_mask = np.zeros(MAX_BARRIERS, dtype=bool)
        self._barrier_duration_config = np.zeros(MAX_BARRIERS, dtype=np.int32)

        # Extract pressure plate info from layout
        self._pressure_plate_positions = np.zeros(
            (MAX_PRESSURE_PLATES, 2), dtype=np.int32
        )
        self._pressure_plate_linked_barrier = np.zeros(
            (MAX_PRESSURE_PLATES, MAX_BARRIERS), dtype=bool
        )
        self._pressure_plate_action_type = np.zeros(MAX_PRESSURE_PLATES, dtype=np.int32)
        self._pressure_plate_active_mask = np.zeros(MAX_PRESSURE_PLATES, dtype=bool)

        for i, (y, x, barrier_targets, action_type) in enumerate(
            layout.pressure_plate_info[:MAX_PRESSURE_PLATES]
        ):
            self._pressure_plate_positions[i] = [y, x]
            for barrier_idx in barrier_targets:
                if 0 <= barrier_idx < MAX_BARRIERS:
                    self._pressure_plate_linked_barrier[i, barrier_idx] = True
            self._pressure_plate_action_type[i] = action_type
            self._pressure_plate_active_mask[i] = True

        for i, (y, x, active) in enumerate(layout.barrier_info[:MAX_BARRIERS]):
            self._barrier_positions[i] = [y, x]
            self._barrier_initial_active[i] = active
            self._barrier_active_mask[i] = True

            # Set duration for each barrier
            if isinstance(barrier_duration, list):
                if i < len(barrier_duration):
                    self._barrier_duration_config[i] = barrier_duration[i]
                else:
                    self._barrier_duration_config[i] = DEFAULT_BARRIER_DURATION
            else:
                self._barrier_duration_config[i] = barrier_duration

        self.config = self._build_config()
        self._step_jit = jax.jit(step_overcooked_v3, static_argnames=("config",))

    def _build_config(self) -> OvercookedV3Config:
        """Build the static configuration object passed through functional logic."""
        return OvercookedV3Config(
            height=self.height,
            width=self.width,
            layout=self.layout,
            num_agents=self.num_agents,
            agents=self.agents,
            action_set=self.action_set,
            observation_type=self.observation_type,
            agent_view_size=self.agent_view_size,
            obs_shape=self.obs_shape,
            max_steps=self.max_steps,
            pot_cook_time=self.pot_cook_time,
            pot_cook_time_range=tuple(
                int(value) for value in self.pot_cook_time_range
            ),
            pot_burn_time=self.pot_burn_time,
            recipe_mode=self.recipe_mode,
            recipe_probs=self.recipe_probs,
            enable_order_queue=self.enable_order_queue,
            max_orders=self.max_orders,
            order_generation_rate=self.order_generation_rate,
            order_expiration_time=self.order_expiration_time,
            order_recipe_encodings=self._order_recipe_encodings,
            enable_item_conveyors=self.enable_item_conveyors,
            enable_player_conveyors=self.enable_player_conveyors,
            enable_moving_walls=self.enable_moving_walls,
            enable_buttons=self.enable_buttons,
            enable_pressure_plates=self.enable_pressure_plates,
            delivery_reward=self.delivery_reward,
            shaped_rewards_enabled=self.shaped_rewards_enabled,
            random_reset=self.random_reset,
            random_agent_positions=self.random_agent_positions,
            possible_recipes=self.possible_recipes,
            enclosed_spaces=self.enclosed_spaces,
            pot_positions=self._pot_positions,
            pot_active_mask=self._pot_active_mask,
            goal_positions=self._goal_positions,
            item_conveyor_positions=self._item_conveyor_positions,
            item_conveyor_directions=self._item_conveyor_directions,
            item_conveyor_active_mask=self._item_conveyor_active_mask,
            player_conveyor_positions=self._player_conveyor_positions,
            player_conveyor_directions=self._player_conveyor_directions,
            player_conveyor_active_mask=self._player_conveyor_active_mask,
            moving_wall_positions=self._moving_wall_positions,
            moving_wall_directions=self._moving_wall_directions,
            moving_wall_active_mask=self._moving_wall_active_mask,
            moving_wall_initial_paused=self._moving_wall_initial_paused,
            moving_wall_bounce=self._moving_wall_bounce,
            button_positions=self._button_positions,
            button_target_idxs=self._button_target_idxs,
            button_target_mask=self._button_target_mask,
            button_action_type=self._button_action_type,
            button_active_mask=self._button_active_mask,
            barrier_positions=self._barrier_positions,
            barrier_initial_active=self._barrier_initial_active,
            barrier_active_mask=self._barrier_active_mask,
            barrier_duration_config=self._barrier_duration_config,
            pressure_plate_positions=self._pressure_plate_positions,
            pressure_plate_linked_barrier=self._pressure_plate_linked_barrier,
            pressure_plate_action_type=self._pressure_plate_action_type,
            pressure_plate_active_mask=self._pressure_plate_active_mask,
        )

    def _validate_recipe_configuration(
        self,
        recipe_mode: str,
        recipe_probs: Optional[Union[List[float], np.ndarray, chex.Array]],
    ) -> None:
        """Validate mode-specific recipe counts, uniqueness, and probabilities."""
        recipes = [
            tuple(int(value) for value in recipe)
            for recipe in self.layout.possible_recipes
        ]
        if len(set(recipes)) != len(recipes):
            raise ValueError("possible_recipes must not contain duplicate recipes")

        num_recipes = len(recipes)
        if recipe_mode == "fixed":
            if num_recipes != 1:
                raise ValueError(
                    "fixed recipe_mode requires exactly one possible recipe"
                )
            if recipe_probs is not None:
                raise ValueError("recipe_probs must be omitted in fixed recipe_mode")
            return

        if recipe_mode == "alternating":
            if num_recipes < 2:
                raise ValueError(
                    "alternating recipe_mode requires at least two possible recipes"
                )
            if recipe_probs is not None:
                raise ValueError(
                    "recipe_probs must be omitted in alternating recipe_mode"
                )
            return

        if recipe_probs is None:
            raise ValueError("random recipe_mode requires recipe_probs")
        self._validate_recipe_probs(recipe_probs)

    def _validate_recipe_probs(
        self, recipe_probs: Optional[Union[List[float], np.ndarray, chex.Array]]
    ) -> None:
        """Validate the configured recipe sampling distribution."""
        if recipe_probs is None:
            return

        messages = []
        try:
            probs = np.asarray(recipe_probs, dtype=np.float32)
        except (TypeError, ValueError):
            raise ValueError("Invalid recipe_probs:\n- recipe_probs must be numeric")

        if probs.ndim != 1:
            messages.append("recipe_probs must be a one-dimensional sequence")
        else:
            num_recipes = self.possible_recipes.shape[0]
            if len(probs) != num_recipes:
                messages.append(
                    "recipe_probs length "
                    f"({len(probs)}) must match possible_recipes length "
                    f"({num_recipes})"
                )

        if not np.all(np.isfinite(probs)):
            messages.append("recipe_probs must contain only finite values")

        if np.any(probs < 0):
            messages.append("recipe_probs must be non-negative")

        if not np.isclose(float(np.sum(probs)), 1.0):
            messages.append("recipe_probs must sum to 1.0")

        if messages:
            formatted_messages = "\n".join(f"- {message}" for message in messages)
            raise ValueError(f"Invalid recipe_probs:\n{formatted_messages}")

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Reset the environment by running the functional reset pipeline."""
        return reset_overcooked_v3(key, self.config)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform a single timestep by forwarding to the functional step pipeline."""
        return self._step_jit(key, state, actions, self.config)

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, float, chex.Array, chex.Array]:
        """Compatibility wrapper for the functional agent action phase."""
        return run_agent_action_phase(key, state, actions, self.config)

    def process_interact(
        self,
        grid: chex.Array,
        agent: Agent,
        all_inventories: jnp.ndarray,
        recipe: int,
        pot_timers: chex.Array,
        pot_positions: chex.Array,
        pot_active_mask: chex.Array,
        pot_cook_time: Optional[chex.Array] = None,
        order_types: Optional[chex.Array] = None,
        order_active_mask: Optional[chex.Array] = None,
    ):
        """Compatibility wrapper for functional interact processing."""
        if order_types is None:
            order_types = jnp.zeros(self.max_orders, dtype=jnp.int32)
        if order_active_mask is None:
            order_active_mask = jnp.zeros(self.max_orders, dtype=jnp.bool_)
        return process_interact(
            grid,
            agent,
            all_inventories,
            recipe,
            pot_timers,
            pot_positions,
            pot_active_mask,
            self.config,
            pot_cook_time,
            order_types,
            order_active_mask,
        )

    def _get_obs_shape(self) -> Tuple[int, ...]:
        """Return the configured observation shape."""
        return calculate_observation_shape(
            self.width,
            self.height,
            self.layout,
            self.observation_type,
            self.agent_view_size,
            self.enable_order_queue,
            self.max_orders,
        )

    def _sample_recipe(self, key: chex.PRNGKey) -> int:
        """Compatibility wrapper for functional recipe sampling."""
        return sample_recipe(key, self.config)

    def _sample_pot_cook_time(self, key: chex.PRNGKey) -> chex.Array:
        """Sample one ready-time duration from the configured inclusive range."""
        return sample_pot_cook_time(key, self.config)

    def _randomize_agent_positions(self, state: State, key: chex.PRNGKey) -> State:
        """Compatibility wrapper for functional agent position randomization."""
        return randomize_agent_positions(state, key, self.config)

    def _randomize_state(self, state: State, key: chex.PRNGKey) -> State:
        """Compatibility wrapper for functional state randomization."""
        return randomize_state(state, key, self.config)

    @staticmethod
    def _is_agent_walkable(static_object, pos, state):
        """Compatibility wrapper for functional walkability checks."""
        return is_agent_walkable(static_object, pos, state)

    @staticmethod
    def _barriers_occupied(agent_ys, agent_xs, barrier_positions, barrier_active_mask):
        """Compatibility wrapper for functional barrier occupancy checks."""
        return barriers_occupied(
            agent_ys, agent_xs, barrier_positions, barrier_active_mask
        )

    def _update_pot_timers(
        self,
        grid: chex.Array,
        pot_timers: chex.Array,
        pot_positions: chex.Array,
        pot_active_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Compatibility wrapper for functional pot timer updates."""
        return update_pot_timers(
            grid, pot_timers, pot_positions, pot_active_mask, self.config
        )

    def _process_item_conveyors(self, state: State) -> State:
        """Compatibility wrapper for functional item conveyor movement."""
        return move_items_on_item_conveyors(state, self.config)

    def _process_player_conveyors(self, state: State) -> State:
        """Compatibility wrapper for functional player conveyor movement."""
        return push_players_on_player_conveyors(state, self.config)

    def _process_barrier_timers(self, state: State) -> State:
        """Compatibility wrapper for functional barrier timer updates."""
        return update_barrier_timers(state, self.config)

    def _process_pressure_plates(self, state: State) -> State:
        """Compatibility wrapper for functional pressure plate updates."""
        return update_pressure_plates(state, self.config)

    def _process_moving_walls(self, state: State) -> State:
        """Compatibility wrapper for functional moving wall updates."""
        return move_moving_walls(state, self.config)

    def _process_order_queue(
        self, state: State, key: chex.PRNGKey
    ) -> Tuple[State, float, chex.Array]:
        """Compatibility wrapper for functional order queue updates."""
        return process_order_queue(state, key, self.config)

    def is_terminal(self, state: State) -> bool:
        """Compatibility wrapper for the functional terminal check."""
        return is_terminal(state, self.config)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Compatibility wrapper for functional observation construction."""
        return get_obs(state, self.config)

    def get_obs_for_type(
        self, state: State, obs_type: ObservationType
    ) -> Dict[str, chex.Array]:
        """Compatibility wrapper for one functional observation encoding."""
        return get_obs_for_type(state, obs_type, self.config)

    def get_obs_default(self, state: State) -> chex.Array:
        """Compatibility wrapper for functional default observations."""
        return get_obs_default(state, self.config)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked V3"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id="") -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set), dtype=jnp.uint32)

    def observation_space(self, agent_id="") -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, self.obs_shape)
