"""Configuration settings for Overcooked V3."""

# Pot timing
POT_COOK_TIME = 80

# Rewards
DELIVERY_REWARD = 20.0    # Base reward for correct delivery
ORDER_EXPIRED_PENALTY = -10.0  # Penalty when order expires

# Order queue defaults
DEFAULT_ORDER_GENERATION_RATE = 0.1
DEFAULT_ORDER_EXPIRATION_TIME = 200
DEFAULT_MAX_ORDERS = 5

# Shaped rewards for intermediate actions
SHAPED_REWARDS = {
    "POT_START_COOKING": 0.3,     # Pot fills and starts cooking (milestone)
}

# Maximum number of pots to track (for fixed array sizes)
MAX_POTS = 4

# Maximum conveyor belt cells
MAX_ITEM_CONVEYORS = 16
MAX_PLAYER_CONVEYORS = 8
