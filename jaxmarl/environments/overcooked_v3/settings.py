"""Configuration settings for Overcooked V3."""

# Pot timing (matching CoGrid defaults)
POT_COOK_TIME = 1        # Steps to cook (CoGrid: cooking_time=90)
POT_BURN_TIME = 10        # Steps in burning window before burned (CoGrid: burning_time=60)

# Rewards
DELIVERY_REWARD = 20.0    # Base reward for correct delivery
# BURN_PENALTY = -5.0       # Penalty when pot burns
ORDER_EXPIRED_PENALTY = -10.0  # Penalty when order expires

# Order queue defaults
DEFAULT_ORDER_GENERATION_RATE = 0.1
DEFAULT_ORDER_EXPIRATION_TIME = 200
DEFAULT_MAX_ORDERS = 5

SHAPED_REWARDS = {
    "INGREDIENT_PICKUP": 3,     # Picking up an ingredient from a pile
    "PLACEMENT_IN_POT": 6,      # Adding correct ingredient to pot
    "SOUP_IN_DISH": 12,          # Picking up cooked soup with plate
    "PLATE_PICKUP": 4,          # Picking up a plate when useful
}


# Maximum number of pots to track (for fixed array sizes)
MAX_POTS = 4

# Maximum conveyor belt cells
MAX_ITEM_CONVEYORS = 16
MAX_PLAYER_CONVEYORS = 8