"""Configuration settings for Overcooked V3."""

# Pot timing (matching v1 cook time of 20 steps)
# Soup is ready when timer reaches POT_BURN_TIME, so cook time = POT_COOK_TIME - POT_BURN_TIME = 20 steps
POT_COOK_TIME = 80        # Initial timer value; cooking finishes after 20 steps (at POT_BURN_TIME)
POT_BURN_TIME = 60        # Steps in burning window before soup burns

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
    "INGREDIENT_PICKUP": 0.1,     # Picking up an ingredient from a pile
    "PLACEMENT_IN_POT": 0.2,      # Adding correct ingredient to pot
    "SOUP_IN_DISH": 0.3,          # Picking up cooked soup with plate
    "PLATE_PICKUP": 0.1,          # Picking up a plate when useful
    "POT_START_COOKING": 0.2,     # Starting to cook a correct recipe
}

# Maximum number of pots to track (for fixed array sizes)
MAX_POTS = 4

# Maximum conveyor belt cells
MAX_ITEM_CONVEYORS = 16
MAX_PLAYER_CONVEYORS = 8
