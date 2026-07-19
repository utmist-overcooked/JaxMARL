# Pot timing (matching CoGrid defaults)
POT_COOK_TIME = 60        # Steps to cook (CoGrid: cooking_time=90)
POT_BURN_TIME = 90        # Steps in burning window before burned (CoGrid: burning_time=60)

# Rewards
DELIVERY_REWARD = 20.0    # Base reward for correct delivery
BURN_PENALTY = -5.0       # Penalty when pot burns
ORDER_EXPIRED_PENALTY = -10.0  # Penalty when order expires

# Order queue defaults
DEFAULT_ORDER_GENERATION_RATE = 0.1
DEFAULT_ORDER_EXPIRATION_TIME = 200
DEFAULT_MAX_ORDERS = 5

# Shaped rewards for intermediate actions
SHAPED_REWARDS = {
    "PLACEMENT_IN_POT": 1.0,      # Adding correct ingredient to pot
    "SOUP_IN_DISH": 10.0,         # Picking up cooked soup with plate
    "PLATE_PICKUP": 1.0,          # Picking up a plate when useful
    "POT_START_COOKING": 5.0,     # Starting to cook a correct recipe (restored to cogridpots_dense value that cooked fresh)
    "HANDOFF_DROP": 0.25,         # Dropping useful item onto a middle handoff counter (RESTORED: key signal for the CTC conveyor handoff)
    "HANDOFF_PICKUP": 0.25,       # Picking useful item up from a middle handoff counter (RESTORED)
    "TASK_PROGRESS": 0.05,        # Moving closer to the current useful object (restored: dense pull empty->plate->pot->goal)
    "TASK_FACING": 0.01,          # Facing a useful object after a movement action
    "INVALID_MOVE": -0.002,       # Trying to move into a blocked cell
    "INGREDIENT_WASTE": -0.004,   # Dropping a wrong-type ingredient onto a conveyor (wastes it)
    "IDLE_PENALTY": -0.001,       # Small penalty for choosing the no-op 'stay' action (discourages freezing)
}

# Maximum number of pots to track (for fixed array sizes)
MAX_POTS = 4

# Maximum conveyor belt cells
MAX_ITEM_CONVEYORS = 16
MAX_PLAYER_CONVEYORS = 8

# Moving walls and buttons
MAX_MOVING_WALLS = 8
MAX_BUTTONS = 8

# Barriers
MAX_BARRIERS = 16
DEFAULT_BARRIER_DURATION = 5  # Default duration for timed barrier deactivation (steps)

# Maximum targets a single button can control. Moving-wall buttons use moving wall
# indexes, barrier buttons use barrier indexes.
MAX_BUTTON_TARGETS = max(MAX_MOVING_WALLS, MAX_BARRIERS)