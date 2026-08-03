"""Configuration settings for Overcooked V3."""

# Pot timing (matching CoGrid defaults)
POT_COOK_TIME = 90
POT_COOK_TIME_RANGE = ()
POT_BURN_TIME = 60

# Prep station timing
CHOP_STAGES = 3
GRILL_COOK_TIME = 15
GRILL_BURN_TIME = 30
BLEND_TIME = 10

# Finite plate supply used only when dish washing is enabled.
DEFAULT_NUM_PLATES = 3

# Rewards
DELIVERY_REWARD = 20.0
BURN_PENALTY = -5.0
ORDER_EXPIRED_PENALTY = -10.0

# Order queue defaults
DEFAULT_ORDER_GENERATION_RATE = 0.1
DEFAULT_ORDER_EXPIRATION_TIME = 200
DEFAULT_MAX_ORDERS = 5

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
    "INGREDIENT_PICKUP": 0.1,
    "PLACEMENT_IN_POT": 0.1,
    "SOUP_IN_DISH": 0.3,
    "PLATE_PICKUP": 0.1,
    "PLATE_PICKUP_DURING_COOKING": 0.0,
    "DISH_TO_GOAL_PROGRESS": 0.0,
    "POT_START_COOKING": 0.2,
    "PREP_PLACEMENT": 0.2,
    "PREP_ACTION": 0.1,
    "PREP_PICKUP": 0.2,
    "DIRTY_PLATE_PICKUP": 0.1,
    "PLATE_WASH": 0.3,
}

MAX_POTS = 4
MAX_ITEM_CONVEYORS = 16
MAX_PLAYER_CONVEYORS = 16
MAX_MOVING_WALLS = 8
MAX_BUTTONS = 16
MAX_PRESSURE_PLATES = 16
MAX_BARRIERS = 16
DEFAULT_BARRIER_DURATION = 5
MAX_BUTTON_TARGETS = max(MAX_MOVING_WALLS, MAX_BARRIERS)
