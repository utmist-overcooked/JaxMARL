"""Layout definitions and parsing for Overcooked V3."""

from jaxmarl.environments.overcooked_v3.common import StaticObject, Direction, ButtonAction
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import itertools


# Standard layouts from Overcooked-AI
cramped_room = """
WWPWW
OA AO
W   W
WBWXW
"""

asymm_advantages = """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""

coord_ring = """
WWWPW
W A P
BAW W
O   W
WOXWW
"""

forced_coord = """
WWWPW
O WAP
OAW W
B W W
WWWXW
"""

counter_circuit = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""

# Layouts with recipe indicators
cramped_room_v2 = """
WWPWW
0A A1
W   R
WBWXW
"""

# New layout with conveyor belts (example)
conveyor_demo = """
WWWPWWW
0A >  X
W  v  W
W  > AW
WWWBWWW
"""

# Player conveyor demo
player_conveyor_demo = """
WWWWWWW
0A ]  X
W     W
W   [ P
WWWBWWW
"""

# Player conveyor loop - 2x2 clockwise loop for testing
# ] pushes right, } pushes down, [ pushes left, { pushes up
player_conveyor_loop = """
WWWWW
W]}AW
W{[0W
WWXWW
"""


# Moving wall demo - wall moves right, button reverses its direction
moving_wall_demo = """
WWWPWWW
0Ae   X
W  !  W
W    AW
WWWBWWW
"""

# Moving wall bounce demo - two walls bouncing back and forth
moving_wall_bounce_demo = """
WWWWPWWWW
0A e   AX
W        W
W  e !   W
WWWWBWWWW
"""


@dataclass
class Layout:
    """Layout definition for Overcooked V3."""
    # Agent positions: list of (x, y) tuples
    agent_positions: List[Tuple[int, int]]

    # height x width grid with static items
    static_objects: np.ndarray

    # Number of unique ingredient types
    num_ingredients: int

    # Possible recipes (list of lists of ingredient indices)
    possible_recipes: Optional[List[List[int]]]

    # Conveyor belt information
    # Item conveyors: list of (y, x, direction) tuples
    item_conveyor_info: List[Tuple[int, int, int]] = field(default_factory=list)

    # Player conveyors: list of (y, x, direction) tuples
    player_conveyor_info: List[Tuple[int, int, int]] = field(default_factory=list)

    # Moving walls: list of (y, x, direction, bounce) tuples
    moving_wall_info: List[Tuple[int, int, int, bool]] = field(default_factory=list)

    # Buttons: list of (y, x, linked_wall_idx, action_type) tuples
    button_info: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        if len(self.agent_positions) == 0:
            raise ValueError("At least one agent position must be provided")
        if self.num_ingredients < 1:
            raise ValueError("At least one ingredient must be available")
        if self.possible_recipes is None:
            self.possible_recipes = self._get_all_possible_recipes(self.num_ingredients)

    @property
    def height(self):
        return self.static_objects.shape[0]

    @property
    def width(self):
        return self.static_objects.shape[1]

    @staticmethod
    def _get_all_possible_recipes(num_ingredients: int) -> List[List[int]]:
        """Get all possible recipes given the number of ingredients."""
        available_ingredients = list(range(num_ingredients)) * 3
        raw_combinations = itertools.combinations(available_ingredients, 3)
        unique_recipes = set(
            tuple(sorted(combination)) for combination in raw_combinations
        )
        return [list(recipe) for recipe in unique_recipes]

    @staticmethod
    def from_string(
        grid,
        possible_recipes=None,
        swap_agents=False,
        moving_wall_bounce=None,
        button_config=None,
    ):
        """Parse a string representation of the layout.

        Symbols:
            W: wall
            A: agent
            X: goal (delivery zone)
            B: plate (bowl) pile
            P: pot location
            R: recipe of the day indicator
            0-9: Ingredient x pile
            ' ' (space): empty cell

            Item conveyor belts (move items):
            >: right
            <: left
            ^: up
            v: down

            Player conveyor belts (push agents):
            ]: right
            [: left
            {: up
            }: down

            Moving walls (move in direction each step):
            n: up
            s: down
            e: right
            w: left (west)

            Buttons (interact to trigger linked wall action):
            !: button (linked to wall by button_config)

        Args:
            grid: ASCII string layout
            possible_recipes: List of recipes, or None for auto-detect
            swap_agents: Reverse agent order
            moving_wall_bounce: List of bools per moving wall (by parse order),
                whether wall bounces when blocked. Default: all False.
            button_config: List of (wall_idx, action_type) per button (by parse
                order). wall_idx is the moving wall index (parse order).
                action_type is a ButtonAction enum value.
                Default: all (0, ButtonAction.TOGGLE_DIRECTION).

        Legacy:
            O: onion pile - will be interpreted as ingredient 0
        """
        rows = grid.split("\n")

        if len(rows[0]) == 0:
            rows = rows[1:]
        if len(rows[-1]) == 0:
            rows = rows[:-1]

        row_lens = [len(row) for row in rows]
        static_objects = np.zeros((len(rows), max(row_lens)), dtype=int)

        char_to_static_item = {
            " ": StaticObject.EMPTY,
            "W": StaticObject.WALL,
            "X": StaticObject.GOAL,
            "B": StaticObject.PLATE_PILE,
            "P": StaticObject.POT,
            "R": StaticObject.RECIPE_INDICATOR,
        }

        # Add ingredient piles 0-9
        for r in range(10):
            char_to_static_item[f"{r}"] = StaticObject.INGREDIENT_PILE_BASE + r

        # Item conveyor belt directions
        item_conveyor_chars = {
            ">": Direction.RIGHT,
            "<": Direction.LEFT,
            "^": Direction.UP,
            "v": Direction.DOWN,
        }

        # Player conveyor belt directions
        player_conveyor_chars = {
            "]": Direction.RIGHT,
            "[": Direction.LEFT,
            "{": Direction.UP,
            "}": Direction.DOWN,
        }

        # Moving wall directions (compass: n=up, s=down, e=east/right, w=west/left)
        moving_wall_chars = {
            "n": Direction.UP,
            "s": Direction.DOWN,
            "e": Direction.RIGHT,
            "w": Direction.LEFT,
        }

        agent_positions = []
        item_conveyor_info = []
        player_conveyor_info = []
        moving_wall_positions = []  # (y, x, direction) before bounce applied
        button_positions = []       # (y, x)

        num_ingredients = 0
        includes_recipe_indicator = False

        for r, row in enumerate(rows):
            c = 0
            while c < len(row):
                char = row[c]

                # Legacy: O -> 0 (onion)
                if char == "O":
                    char = "0"

                if char == "A":
                    agent_pos = (c, r)
                    agent_positions.append(agent_pos)
                    static_objects[r, c] = StaticObject.EMPTY
                elif char in item_conveyor_chars:
                    static_objects[r, c] = StaticObject.ITEM_CONVEYOR
                    direction = item_conveyor_chars[char]
                    item_conveyor_info.append((r, c, direction))
                elif char in player_conveyor_chars:
                    static_objects[r, c] = StaticObject.PLAYER_CONVEYOR
                    direction = player_conveyor_chars[char]
                    player_conveyor_info.append((r, c, direction))
                elif char in moving_wall_chars:
                    static_objects[r, c] = StaticObject.MOVING_WALL
                    direction = moving_wall_chars[char]
                    moving_wall_positions.append((r, c, direction))
                elif char == "!":
                    static_objects[r, c] = StaticObject.BUTTON
                    button_positions.append((r, c))
                else:
                    obj = char_to_static_item.get(char, StaticObject.EMPTY)
                    static_objects[r, c] = obj

                    if StaticObject.is_ingredient_pile(obj):
                        ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                        num_ingredients = max(num_ingredients, ingredient_idx + 1)

                    if obj == StaticObject.RECIPE_INDICATOR:
                        includes_recipe_indicator = True

                c += 1

        # Validation
        if possible_recipes is not None:
            if not isinstance(possible_recipes, list):
                raise ValueError("possible_recipes must be a list")
            if not all(isinstance(recipe, list) for recipe in possible_recipes):
                raise ValueError("possible_recipes must be a list of lists")
            if not all(len(recipe) == 3 for recipe in possible_recipes):
                raise ValueError("All recipes must be of length 3")
        elif not includes_recipe_indicator:
            raise ValueError(
                "Layout does not include a recipe indicator, a fixed recipe must be provided"
            )

        if swap_agents:
            agent_positions = agent_positions[::-1]

        # Ensure at least one ingredient type
        if num_ingredients == 0:
            num_ingredients = 1

        # Build moving wall info with bounce config
        if moving_wall_bounce is None:
            moving_wall_bounce = [False] * len(moving_wall_positions)
        if len(moving_wall_bounce) != len(moving_wall_positions):
            raise ValueError(
                f"moving_wall_bounce length ({len(moving_wall_bounce)}) must match "
                f"number of moving walls ({len(moving_wall_positions)})"
            )
        moving_wall_info = [
            (y, x, direction, bounce)
            for (y, x, direction), bounce in zip(moving_wall_positions, moving_wall_bounce)
        ]

        # Build button info with config
        if button_config is None:
            button_config = [(0, ButtonAction.TOGGLE_DIRECTION)] * len(button_positions)
        if len(button_config) != len(button_positions):
            raise ValueError(
                f"button_config length ({len(button_config)}) must match "
                f"number of buttons ({len(button_positions)})"
            )
        button_info = [
            (y, x, wall_idx, action_type)
            for (y, x), (wall_idx, action_type) in zip(button_positions, button_config)
        ]

        layout = Layout(
            agent_positions=agent_positions,
            static_objects=static_objects,
            num_ingredients=num_ingredients,
            possible_recipes=possible_recipes,
            item_conveyor_info=item_conveyor_info,
            player_conveyor_info=player_conveyor_info,
            moving_wall_info=moving_wall_info,
            button_info=button_info,
        )

        return layout


# Pre-defined layouts
overcooked_v3_layouts = {
    # Original Overcooked-AI layouts
    "cramped_room": Layout.from_string(
        cramped_room, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),
    "asymm_advantages": Layout.from_string(
        asymm_advantages, possible_recipes=[[0, 0, 0]]
    ),
    "coord_ring": Layout.from_string(coord_ring, possible_recipes=[[0, 0, 0]]),
    "forced_coord": Layout.from_string(forced_coord, possible_recipes=[[0, 0, 0]]),
    "counter_circuit": Layout.from_string(
        counter_circuit, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),

    # V2-style layouts with recipe indicators
    "cramped_room_v2": Layout.from_string(cramped_room_v2),

    # Demo layouts with conveyors
    "conveyor_demo": Layout.from_string(
        conveyor_demo, possible_recipes=[[0, 0, 0]]
    ),
    "player_conveyor_demo": Layout.from_string(
        player_conveyor_demo, possible_recipes=[[0, 0, 0]]
    ),

    # 2x2 clockwise conveyor loop for testing
    "player_conveyor_loop": Layout.from_string(
        player_conveyor_loop, possible_recipes=[[0, 0, 0]]
    ),

    # Moving wall demos
    "moving_wall_demo": Layout.from_string(
        moving_wall_demo,
        possible_recipes=[[0, 0, 0]],
        button_config=[(0, ButtonAction.TOGGLE_DIRECTION)],
    ),
    "moving_wall_bounce_demo": Layout.from_string(
        moving_wall_bounce_demo,
        possible_recipes=[[0, 0, 0]],
        moving_wall_bounce=[True, True],
        button_config=[(1, ButtonAction.TOGGLE_PAUSE)],
    ),
}
