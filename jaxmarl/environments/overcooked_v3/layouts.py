"""Layout definitions and parsing for Overcooked V3.

DESIGN NOTES:
- don't make item conveyor belts / player conveyor belts move things to the same destination - this will cause race conditions and maybe make the items disappear.
"""

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

race_against_the_clock = """
XWWWWWWWWWW 2
            1
 WWWWWWWWWW 0
AW        WWW
PW          W
AW        WWW
 WWWWWWWWWW 0
            1
XWWWWWWWWWW 2
"""

maze_conveyor_hell  = """
01 W   W   v
A  W W W  Wv
vW W W W  Wv
vW   W    Wv
vWWWWWWWW Wv
vW>>>>>>> Wv
vW WWWWWWPWv
A       WXW 
B
"""

coordinated_temporal_conveyor = """
>>>>>>vW   X
      vW  A 
     WvW    
     Wv    B
  A  WvWWWWW
01   Wv>>>>>
"""

general_conveyor_level_1 = """
012    P
       W
       W
]]]]]]]]
[[[[[[[[
       W
       W
BX     P
"""

general_conveyor_level_2 = """
W W   1
0  WW WW  P
2A  ]]]   P
 A  W W    
   WW WW   
    [[[   B
   WW WW  B
   XW W    
   XW W
"""

general_conveyor_level_3 = """
A 01WWW
A]]]]}2
W{WWW}W
W{[[[[W
WBWWPWW
"""

middle_conveyor = """
WWWWW^WWWWW
WW  W^W  WW
WW AW^WA WW
W1   ^   PW
WW  W^W  WW
WW BW^WB WW
W0   ^   XW
WW  W^W  WW
WW  W^W  WW
WWWWW^WWWWW
"""


follow_the_leader = """
WWWWWWWW
WWB1  WW
W0AWA PW
W  W  WW
WWWW  XW
W     WW
WWWWWWWW
"""

around_the_island = """
WW0W1WWWWW
B        W
W  A     W
WWWWWWW  X
W  A     W
W        W
WWWPWWWWWW
"""

single_file = """
WBWWPWW
W A A W
W WWW W
X     W
WW1W0WW
"""


# Moving wall demo - wall moves right, button reverses its direction
moving_wall_demo = """
WWWPWWW
0As   X
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

# Barrier demo - togglable barriers that block all directions
barrier_demo = """
WWWPWWW
0A #  X
W  #  W
W    AW
WWWBWWW
"""

# Timed barrier demo - button deactivates barrier temporarily
timed_barrier_demo = """
WWWPWWW
0A #  X
W ! ! W
W  # AW
WWWBWWW
"""

# Pressure Plate Sample Level - step on plate to toggle multiple walls
pressure_plate_demo = """
WWWWWWW
0 #_AWW
WWWW WW
P #_  X
WWWW WW
1 #_AWW
WWWWWWW
"""

# pressure gated conveyor access
pressure_gated_conveyor_access = """
WWWWWWBWW
WA  #   W
W  _W1v0W
WX  WWvWW
W  _WWvWW
WA  #  PW
WWWWWWRWW
"""

# pressure gated circuit
pressure_gated_circuit = """
WWWWWWWWW
W   #   W
W PWWW0 W
W W _ W W
W#W_A_W#W
W W _ W W
W XWWWB W
W   #  AW
WWWWWWRWW
"""

pressure_gated_zones = """
WWWWWWW0W
WP  #   1
W  _R_  W
W _ W _ W
W#RWWWR#W
W _ W _ W
W  _R_ AW
WB  # AXW
WWWWWWWWW
"""

twin_movement = """
WWWWWWWWWWPWWWW
WWWWWWWWWW#WWWW
WWW_WWWW#   #WW
WW___WWW W#W XW
W__A__B# #A# #0
WW___WWW W#W XW
WWW_WWWW#   #WW
WWWWWWWWWW#WWWW
WWWWWWWWWW1WWWW
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

    # Pressure Plates: list of (y, x, target_barrier_idx, action_type) tuples
    pressure_plate_info: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Barriers: list of (y, x, active) tuples
    barrier_info: List[Tuple[int, int, bool]] = field(default_factory=list)

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
        barrier_config=None,
        pressure_plate_config=None,
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

            Pressure Plate (interact to trigger linked wall action):
            _: pressure plate (linked to wall by button_config, triggers when agent overlaps)

            Barriers (togglable blocking tiles):
            #: barrier (blocks all movement when active)

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
            barrier_config: List of bools per barrier (by parse order),
                whether barrier is initially active. Default: all False.

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
            "#": StaticObject.BARRIER,
            "_": StaticObject.PRESSURE_PLATE,
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
        moving_wall_positions = []      # (y, x, direction)
        button_positions = []           # (y, x)
        barrier_positions = []          # (y, x)
        pressure_plate_positions = []   # (y, x)

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
                elif char == "_":
                    static_objects[r, c] = StaticObject.PRESSURE_PLATE
                    pressure_plate_positions.append((r, c))
                elif char == "#":
                    static_objects[r, c] = StaticObject.BARRIER
                    barrier_positions.append((r, c))
                else:
                    obj = char_to_static_item.get(char, StaticObject.EMPTY)
                    static_objects[r, c] = obj

                    if StaticObject.is_ingredient_pile(obj):
                        ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                        num_ingredients = max(num_ingredients, ingredient_idx + 1)

                    if obj == StaticObject.RECIPE_INDICATOR:
                        includes_recipe_indicator = True

                c += 1

        # Validation for recipes 
        # NOTE: possible_recipes is a list of lists of ingredient indices, max 3 ingredients per recipe - if no recipe indicator, we just auto-gen all possible combinations. Otherwise we just take the possible_recipes specified in layout. 
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

        # Build barrier info with config
        if barrier_config is None:
            barrier_config = [False] * len(barrier_positions)
        if len(barrier_config) != len(barrier_positions):
            raise ValueError(
                f"barrier_config length ({len(barrier_config)}) must match "
                f"number of barriers ({len(barrier_positions)})"
            )
        barrier_info = [
            (y, x, active)
            for (y, x), active in zip(barrier_positions, barrier_config)
        ]

        # Build pressure plate info with config
        if pressure_plate_config is None:
            pressure_plate_config = [(0, ButtonAction.TOGGLE_BARRIER)] * len(pressure_plate_positions)
        if len(pressure_plate_config) != len(pressure_plate_positions):
            raise ValueError(
                f"pressure_plate_config length ({len(pressure_plate_config)}) must match "
                f"number of pressure plates ({len(pressure_plate_positions)})"
            )
        pressure_plate_info = [
            (y, x, barrier_idx, action_type)
            for (y, x), (barrier_idx, action_type) in zip(pressure_plate_positions, pressure_plate_config)
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
            pressure_plate_info=pressure_plate_info,
            barrier_info=barrier_info,
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

    # Barrier demo
    "barrier_demo": Layout.from_string(
        barrier_demo,
        possible_recipes=[[0, 0, 0]],
        barrier_config=[False, True],  # First barrier off, second barrier on initially
    ),

    # Timed barrier demo with button
    "timed_barrier_demo": Layout.from_string(
        timed_barrier_demo,
        possible_recipes=[[0, 0, 0]],
        barrier_config=[True, True],  # Barrier starts active
        button_config=[(0, ButtonAction.TIMED_BARRIER), (1, ButtonAction.TIMED_BARRIER)],  # Button controls barrier 0 with timed toggle
    ),

    # Pressure Plate Demo:
    "pressure_plate_demo": Layout.from_string(
        pressure_plate_demo,
        possible_recipes=[[0, 0, 0]],
        pressure_plate_config=[
            (1, ButtonAction.TOGGLE_BARRIER),
            (2, ButtonAction.TOGGLE_BARRIER),
            (0, ButtonAction.TOGGLE_BARRIER),
        ],
        barrier_config=[
            True,  # Barrier 0 (Top lane) - initially active
            True,  # Barrier 1 (Middle lane) - initially active
            True,  # Barrier 2 (Bottom lane) - initially active
        ],
    ),

    # Pressure Plate Demo:
    "pressure_gated_conveyor_access": Layout.from_string(
        pressure_gated_conveyor_access,
        possible_recipes=[[0, 0, 0]],
        pressure_plate_config=[
            (0, ButtonAction.TOGGLE_BARRIER),
            (1, ButtonAction.TOGGLE_BARRIER)
        ],
        barrier_config=[
            True,  # Barrier 0 (Top) - initially active
            True,  # Barrier 1 (Bottom) - initially active
        ],
    ),

    # Pressure Plate Demo:
    "pressure_gated_circuit": Layout.from_string(
        pressure_gated_circuit,
        possible_recipes=[[0, 0, 0]],
        pressure_plate_config=[
            (0, ButtonAction.TOGGLE_BARRIER),
            (1, ButtonAction.TOGGLE_BARRIER),
            (2, ButtonAction.TOGGLE_BARRIER),
            (3, ButtonAction.TOGGLE_BARRIER)
        ],
        barrier_config=[
            True,  # Barrier 0 (?) - initially active
            True,  # Barrier 1 (?) - initially active
            True,  # Barrier 2 (?) - initially active
            True  # Barrier 3 (?) - initially active
        ],
    ),

    # Pressure Plate Demo:
    "pressure_gated_zones": Layout.from_string(
        pressure_gated_zones,
        possible_recipes=[[0, 0, 0]],
        pressure_plate_config=[
            (0, ButtonAction.TOGGLE_BARRIER),
            (0, ButtonAction.TOGGLE_BARRIER),
            (1, ButtonAction.TOGGLE_BARRIER),
            (2, ButtonAction.TOGGLE_BARRIER),
            (1, ButtonAction.TOGGLE_BARRIER),
            (2, ButtonAction.TOGGLE_BARRIER),
            (3, ButtonAction.TOGGLE_BARRIER),
            (3, ButtonAction.TOGGLE_BARRIER)
        ],
        barrier_config=[
            True,  # Barrier 0 (?) - initially active
            True,  # Barrier 1 (?) - initially active
            True,  # Barrier 2 (?) - initially active
            True  # Barrier 3 (?) - initially active
        ],
    ),

    # Pressure Plate Demo:
    "twin_movement": Layout.from_string(
        twin_movement,
        possible_recipes=[[0, 0, 0]],
        pressure_plate_config=[
            (0, ButtonAction.TOGGLE_BARRIER),
            (1, ButtonAction.TOGGLE_BARRIER),
            (3, ButtonAction.TOGGLE_BARRIER),
            (2, ButtonAction.TOGGLE_BARRIER),
            (4, ButtonAction.TOGGLE_BARRIER),
            (5, ButtonAction.TOGGLE_BARRIER),
            (6, ButtonAction.TOGGLE_BARRIER),
            (7, ButtonAction.TOGGLE_BARRIER),
            (9, ButtonAction.TOGGLE_BARRIER),
            (8, ButtonAction.TOGGLE_BARRIER),
            (10, ButtonAction.TOGGLE_BARRIER),
            (11, ButtonAction.TOGGLE_BARRIER)
        ],
        barrier_config=[
            True,  # Barrier 0 (?) - initially active
            True,  # Barrier 1 (?) - initially active
            True,  # Barrier 2 (?) - initially active
            True,  # Barrier 3 (?) - initially active
            True,  # Barrier 4 (?) - initially active
            True,  # Barrier 5 (?) - initially active
            True,  # Barrier 6 (?) - initially active
            True,  # Barrier 7 (?) - initially active
            True,  # Barrier 8 (?) - initially active
            True,  # Barrier 9 (?) - initially active
            True,  # Barrier 10 (?) - initially active
            True   # Barrier 11 (?) - initially active
        ],
    ),



    "middle_conveyor": Layout.from_string(
        middle_conveyor, possible_recipes=[[0, 0, 0]],
    ),

    "follow_the_leader": Layout.from_string(
        follow_the_leader, possible_recipes=[[0, 0, 0]],
    ),

    "around_the_island": Layout.from_string(
        around_the_island, possible_recipes=[[0, 0, 0]],
    ),

    "single_file": Layout.from_string(
        single_file, possible_recipes=[[0, 0, 0]],
    ),

}
