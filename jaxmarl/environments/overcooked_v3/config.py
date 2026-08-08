"""Static configuration passed into Overcooked V3 functional core logic."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

@dataclass(eq=False)
class OvercookedV3Config:
    """Static environment configuration used by JAX-traceable step functions."""

    height: int
    width: int
    layout: Any
    num_agents: int
    agents: List[str]
    action_set: Any
    observation_type: Any
    agent_view_size: Optional[int]
    obs_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]
    max_steps: int
    pot_cook_time: int
    pot_cook_time_range: Tuple[int, ...]
    pot_burn_time: int
    enable_random_recipe: bool
    recipe_probs: Any
    enable_order_queue: bool
    max_orders: int
    order_generation_rate: float
    order_expiration_time: int
    order_queue_mode: str
    order_recipe_encodings: Any
    enable_item_conveyors: bool
    enable_player_conveyors: bool
    enable_moving_walls: bool
    enable_buttons: bool
    enable_pressure_plates: bool
    delivery_reward: float
    shaped_rewards_enabled: bool
    random_reset: bool
    random_agent_positions: bool
    possible_recipes: Any
    enclosed_spaces: Any
    pot_positions: Any
    pot_active_mask: Any
    goal_positions: Any
    item_conveyor_positions: Any
    item_conveyor_directions: Any
    item_conveyor_active_mask: Any
    player_conveyor_positions: Any
    player_conveyor_directions: Any
    player_conveyor_active_mask: Any
    moving_wall_positions: Any
    moving_wall_directions: Any
    moving_wall_active_mask: Any
    moving_wall_initial_paused: Any
    moving_wall_bounce: Any
    button_positions: Any
    button_target_idxs: Any
    button_target_mask: Any
    button_action_type: Any
    button_active_mask: Any
    barrier_positions: Any
    barrier_initial_active: Any
    barrier_active_mask: Any
    barrier_duration_config: Any
    pressure_plate_positions: Any
    pressure_plate_linked_barrier: Any
    pressure_plate_action_type: Any
    pressure_plate_active_mask: Any
