"""Configuration settings for Overcooked V3."""

# Pot timing.
POT_COOK_TIME = 20        # Full pot becomes cooked after exactly 20 env steps
POT_COOK_TIME_RANGE = ()  # Optional inclusive [min, max] range for random ready times
POT_BURN_TIME = 40        # Cooked soup expires/burns after this many ready steps

# Prep station timing.
CHOP_STAGES = 3           # Interacts needed on a cutting board to finish chopping
GRILL_COOK_TIME = 15      # Steps for a raw item on the grill to finish grilling
GRILL_BURN_TIME = 30      # Grilled item burns after this many ready steps (0 = never)
BLEND_TIME = 10           # Steps a started blender needs to finish

# Dish washing. Only used when the env runs with dish washing enabled; otherwise
# the plate pile is infinite and no plate is ever consumed.
DEFAULT_NUM_PLATES = 3    # Clean plates the kitchen starts with

# Rewards
DELIVERY_REWARD = 20.0    # Base reward for correct delivery
BURN_PENALTY = -5.0       # Penalty when a cooked pot burns before pickup
ORDER_EXPIRED_PENALTY = -10.0  # Penalty when order expires

# Order queue defaults
DEFAULT_ORDER_GENERATION_RATE = 0.1
DEFAULT_ORDER_EXPIRATION_TIME = 200
DEFAULT_MAX_ORDERS = 5

# Per-agent event counters emitted in step_env's info dict, in the order they
# occupy columns of the event_metrics array threaded through the step pipeline.
EVENT_NAMES = (
    "pot_start_cooking",
    "pot_placement",
    "pickup",
    "drop",
    "dish_pickup",
    "dish_to_goal_progress",
    "delivery",
    "pot_burn",
    "prep_placement",
    "prep_action",
    "prep_pickup",
    "prep_burn",
    "dirty_pickup",
    "plate_wash",
    "plate_return",
)

# Shaped rewards for intermediate actions
SHAPED_REWARDS = {
    # Base kitchen loop (macro/comm-tuned magnitudes from the macro-with-comm line).
    "INGREDIENT_PICKUP": 1.0,     # Picking up an ingredient from a pile
    "PLACEMENT_IN_POT": 1.0,      # Adding correct ingredient to pot
    "SOUP_IN_DISH": 10.0,         # Picking up cooked soup with plate
    "PLATE_PICKUP": 1.0,          # Picking up a plate when useful
    "PLATE_PICKUP_DURING_COOKING": 0.0,  # Disabled ablation; keep key for compatibility
    "DISH_TO_GOAL_PROGRESS": 0.0, # Logged only; no Euclidean distance reward
    "POT_START_COOKING": 5.0,     # Starting to cook a correct recipe
    "HANDOFF_DROP": 0.25,         # Dropping useful item onto a middle handoff counter
    "HANDOFF_PICKUP": 0.25,       # Picking useful item up from a middle handoff counter
    "TASK_PROGRESS": 0.02,        # Moving closer to the current useful object
    "TASK_FACING": 0.01,          # Facing a useful object after a movement action
    "INVALID_MOVE": -0.002,       # Trying to move into a blocked cell
    # Multi-stage prep and dish-washing stages. The feat branch weighted these
    # relative to its own PLACEMENT_IN_POT=0.2 / PLATE_PICKUP=0.1 anchors; those
    # anchors are 5x and 10x larger here, so the prep weights are rescaled by the
    # same factors to preserve the intended relative shaping.
    "PREP_PLACEMENT": 1.0,        # Placing an ingredient on a prep station
    "PREP_ACTION": 0.5,           # Working a prep station (chop/grill/blend)
    "PREP_PICKUP": 1.0,           # Collecting the prepared ingredient
    "DIRTY_PLATE_PICKUP": 1.0,    # Picking up a dirty plate for washing
    "PLATE_WASH": 3.0,            # Washing a dirty plate back into a clean plate
}

# Every distinct reward source an agent can collect in a step, for per-type
# breakdowns (e.g. reward-hacking diagnostics). DELIVERY is the sparse base
# reward; the rest mirror SHAPED_REWARDS keys.
REWARD_COMPONENT_KEYS = (
    "DELIVERY",
    "BURN_PENALTY",
    "ORDER_EXPIRED_PENALTY",
    "SOUP_IN_DISH",
    "PLACEMENT_IN_POT",
    "HANDOFF_DROP",
    "HANDOFF_PICKUP",
    "POT_START_COOKING",
    "PLATE_PICKUP",
    "INGREDIENT_PICKUP",
    "PREP_PLACEMENT",
    "PREP_ACTION",
    "PREP_PICKUP",
    "DIRTY_PLATE_PICKUP",
    "PLATE_WASH",
    "TASK_PROGRESS",
    "TASK_FACING",
    "INVALID_MOVE",
)

# Maximum number of pots to track (for fixed array sizes)
MAX_POTS = 4

# Maximum conveyor belt cells
MAX_ITEM_CONVEYORS = 16
MAX_PLAYER_CONVEYORS = 8

# Moving walls, pressure plates and buttons
MAX_MOVING_WALLS = 8
MAX_BUTTONS = 16
MAX_PRESSURE_PLATES = 16

# Barriers
MAX_BARRIERS = 16
DEFAULT_BARRIER_DURATION = 5  # Default duration for timed barrier deactivation (steps)

# Maximum targets a single button can control. Moving-wall buttons use moving wall
# indexes, barrier buttons use barrier indexes.
MAX_BUTTON_TARGETS = max(MAX_MOVING_WALLS, MAX_BARRIERS)
