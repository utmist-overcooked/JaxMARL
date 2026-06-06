"""Configuration settings for Overcooked V3."""

# Pot timing.
POT_COOK_TIME = 20        # Full pot becomes cooked after exactly 20 env steps
POT_BURN_TIME = 40        # Cooked soup expires/burns after this many ready steps

# Rewards
DELIVERY_REWARD = 20.0    # Base reward for correct delivery
BURN_PENALTY = -5.0       # Penalty when a cooked pot burns before pickup
ORDER_EXPIRED_PENALTY = -10.0  # Penalty when order expires

# Order queue defaults
DEFAULT_ORDER_GENERATION_RATE = 0.1
DEFAULT_ORDER_EXPIRATION_TIME = 200
DEFAULT_MAX_ORDERS = 5

# Shaped rewards for intermediate actions
SHAPED_REWARDS = {
    "INGREDIENT_PICKUP": 0.1,     # Picking up an ingredient from a pile
    "PLACEMENT_IN_POT": 0.2,      # Adding correct ingredient to pot
    "SOUP_IN_DISH": 0.6,          # Picking up cooked soup with plate
    "PLATE_PICKUP": 0.1,          # Picking up a plate when useful
    "PLATE_PICKUP_DURING_COOKING": 0.0,  # Disabled ablation; keep key for compatibility
    "DISH_TO_GOAL_PROGRESS": 0.0, # Logged only; no Euclidean distance reward
    "POT_START_COOKING": 0.2,     # Starting to cook a correct recipe
}

# Maximum number of pots to track (for fixed array sizes)
MAX_POTS = 4

# Maximum conveyor belt cells
MAX_ITEM_CONVEYORS = 16
MAX_PLAYER_CONVEYORS = 8
