"""State containers and observation type definitions for Overcooked V3."""

from enum import Enum

import chex

from jaxmarl.environments.overcooked_v3.common import Agent

class ObservationType(str, Enum):
    """Available observation encodings for Overcooked V3 agents."""

    DEFAULT = "default"
    FEATURIZED = "featurized"

@chex.dataclass
class State:
    """Environment state for Overcooked V3."""

    agents: Agent

    # Grid: height x width x 3 channels
    # Channel 0: static objects
    # Channel 1: dynamic items (plates, ingredients, soups)
    # Channel 2: extra info (pot timers, conveyor directions)
    grid: chex.Array

    # Pot state (fixed size arrays for JIT compatibility)
    # pot_positions stores (y, x) for each pot, pot_active_mask indicates valid pots
    pot_positions: chex.Array  # [max_pots, 2] - (y, x) positions
    pot_cooking_timer: chex.Array  # [max_pots] - countdown to cooked (0 when idle/cooked)
    pot_cook_durations: chex.Array  # [max_pots] - sampled steps until ready
    pot_active_mask: chex.Array  # [max_pots] - bool, which pot slots are valid

    # Order queue state (optional feature)
    order_types: chex.Array  # [max_orders] - SoupType enum values
    order_expirations: chex.Array  # [max_orders] - steps remaining
    order_active_mask: chex.Array  # [max_orders] - bool, which order slots are valid

    # Item conveyor state
    item_conveyor_positions: chex.Array  # [max_item_conveyors, 2] - (y, x)
    item_conveyor_directions: chex.Array  # [max_item_conveyors] - Direction enum
    item_conveyor_active_mask: chex.Array  # [max_item_conveyors] - bool

    # Player conveyor state
    player_conveyor_positions: chex.Array  # [max_player_conveyors, 2] - (y, x)
    player_conveyor_directions: chex.Array  # [max_player_conveyors] - Direction enum
    player_conveyor_active_mask: chex.Array  # [max_player_conveyors] - bool

    # Moving wall state
    moving_wall_positions: chex.Array  # [max_moving_walls, 2] - (y, x)
    moving_wall_directions: chex.Array  # [max_moving_walls] - Direction enum
    moving_wall_active_mask: chex.Array  # [max_moving_walls] - bool
    moving_wall_paused: chex.Array  # [max_moving_walls] - bool
    moving_wall_bounce: chex.Array  # [max_moving_walls] - bool

    # Button state
    button_positions: chex.Array  # [max_buttons, 2] - (y, x)
    button_target_idxs: chex.Array  # [max_buttons, max_button_targets]
    button_target_mask: chex.Array  # [max_buttons, max_button_targets] - bool
    button_action_type: chex.Array  # [max_buttons] - ButtonAction enum
    button_active_mask: chex.Array  # [max_buttons] - bool
    button_toggled: chex.Array  # [max_buttons] - bool (current toggle state)

    # Barrier state
    barrier_positions: chex.Array  # [max_barriers, 2] - (y, x)
    barrier_active: chex.Array  # [max_barriers] - bool (blocks when True)
    barrier_active_mask: chex.Array  # [max_barriers] - bool (which slots are valid)
    barrier_timer: chex.Array  # [max_barriers] - steps until reactivation (0 means permanent state)
    barrier_duration: chex.Array  # [max_barriers] - configured duration for timed deactivation

    # Pressure Plate State
    pressure_plate_positions: chex.Array  # [max_pressure_plates, 2] - (y, x)
    # [max_pressure_plates, max_barriers] - linked barrier mask
    pressure_plate_linked_barrier: chex.Array
    pressure_plate_action_type: chex.Array  # [max_pressure_plates] - ButtonAction enum
    pressure_plate_active_mask: chex.Array  # [max_pressure_plates] - bool (which slots are valid)
    pressure_plate_toggled: chex.Array  # [max_pressure_plates] - bool (currently pressed)

    # Episode state
    time: chex.Array
    terminal: bool
    recipe: int  # Current target recipe (bit-encoded)

    # Delivery tracking
    new_correct_delivery: bool = False
