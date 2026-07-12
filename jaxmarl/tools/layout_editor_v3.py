#!/usr/bin/env python3
"""Interactive visual level editor for Overcooked V3.

Controls:
  Left Click: Place selected object
  Right Click: Erase object

  Keyboard shortcuts:
    W = Wall          P = Pot           B = Plate Pile
    X = Delivery      A = Agent         R = Recipe Indicator
    0-9 = Ingredients E = Erase
    Q = Button        K = Barrier       L = Pressure Plate

    Ctrl+Z = Undo     Ctrl+Y = Redo
    Ctrl+N = New      Ctrl+O = Open     Ctrl+S = Save
    Ctrl+E = Export   Ctrl+T = Test Play

  Menu clicks:
    New, Load, Export, Test, Validate, Quit

Button / Pressure Plate / Barrier Wiring:
  - Place a Barrier tile first (it gets an auto-assigned index shown in brackets).
  - Place a Button or Pressure Plate, then use the Wiring Panel (right panel)
    to link it to one or more Barriers and choose the action type.
  - The wiring panel appears when you hover a Button or Pressure Plate cell.
"""

import sys
import json
import pygame
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path

# Lazy-load JaxMARL dependencies to avoid requiring full jaxmarl installation
DEPENDENCIES_AVAILABLE = False
StaticObject = None
Direction = None
ButtonAction = None
Layout = None
overcooked_v3_layouts = None
OvercookedV3Visualizer = None

# Fallback static object IDs (match Overcooked V3 StaticObject values)
FALLBACK_WALL = 1
FALLBACK_GOAL = 4
FALLBACK_POT = 5
FALLBACK_RECIPE = 6
FALLBACK_PLATE_PILE = 9
FALLBACK_INGREDIENT_BASE = 10
FALLBACK_ITEM_CONVEYOR = 20
FALLBACK_PLAYER_CONVEYOR = 21
FALLBACK_MOVING_WALL = 22
FALLBACK_BUTTON = 23
FALLBACK_BARRIER = 24
FALLBACK_PRESSURE_PLATE = 25

# ButtonAction values. Keep these in sync with overcooked_v3.common.ButtonAction.
BUTTON_ACTION_TOGGLE_PAUSE = 0
BUTTON_ACTION_TOGGLE_DIRECTION = 1
BUTTON_ACTION_TOGGLE_BOUNCE = 2
BUTTON_ACTION_TRIGGER_MOVE = 3
BUTTON_ACTION_TOGGLE_BARRIER = 4
BUTTON_ACTION_TIMED_BARRIER = 5
DEFAULT_WIRE_ACTION = BUTTON_ACTION_TOGGLE_BARRIER

# Fallback ButtonAction values
FALLBACK_BUTTON_ACTIONS = {
    BUTTON_ACTION_TOGGLE_PAUSE: "TOGGLE_PAUSE",
    BUTTON_ACTION_TOGGLE_DIRECTION: "TOGGLE_DIRECTION",
    BUTTON_ACTION_TOGGLE_BOUNCE: "TOGGLE_BOUNCE",
    BUTTON_ACTION_TRIGGER_MOVE: "TRIGGER_MOVE",
    BUTTON_ACTION_TOGGLE_BARRIER: "TOGGLE_BARRIER",
    BUTTON_ACTION_TIMED_BARRIER: "TIMED_BARRIER",
}


def _load_jaxmarl_deps():
    """Load JaxMARL dependencies on first use."""
    global DEPENDENCIES_AVAILABLE, StaticObject, Direction, ButtonAction
    global Layout, overcooked_v3_layouts, OvercookedV3Visualizer

    if DEPENDENCIES_AVAILABLE:
        return

    try:
        from jaxmarl.environments.overcooked_v3.common import (
            StaticObject as SO,
            Direction as Dir,
            ButtonAction as BA,
        )
        from jaxmarl.environments.overcooked_v3.layouts import (
            Layout as L,
            overcooked_v3_layouts as layouts,
        )
        from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer as Viz

        StaticObject = SO
        Direction = Dir
        ButtonAction = BA
        Layout = L
        overcooked_v3_layouts = layouts
        OvercookedV3Visualizer = Viz
        DEPENDENCIES_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import JaxMARL dependencies: {e}")
        print("Editor will run in limited mode")
        DEPENDENCIES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (180, 180, 180)
COLOR_DARK_GRAY = (64, 64, 64)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 100, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE = (255, 165, 0)
COLOR_PURPLE = (160, 32, 240)
COLOR_CYAN = (0, 255, 255)
COLOR_BROWN = (139, 69, 19)
COLOR_DARK_GREEN = (0, 150, 0)
COLOR_PINK = (255, 105, 180)
COLOR_TEAL = (0, 180, 180)
COLOR_MAROON = (180, 30, 30)
COLOR_LIME = (180, 255, 0)

INGREDIENT_COLORS = [
    COLOR_YELLOW,
    COLOR_RED,
    COLOR_DARK_GREEN,
    COLOR_CYAN,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_BLUE,
    COLOR_PINK,
    COLOR_BROWN,
    COLOR_WHITE,
]

AGENT_COLORS = [COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_PURPLE, COLOR_YELLOW, COLOR_ORANGE]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
TILE_SIZE = 48
TOOLBAR_WIDTH = 250
INFO_PANEL_WIDTH = 300
TOP_MENU_HEIGHT = 40
MIN_GRID_WIDTH = 5
MIN_GRID_HEIGHT = 4
MAX_GRID_WIDTH = 20
MAX_GRID_HEIGHT = 20
DEFAULT_GRID_WIDTH = 7
DEFAULT_GRID_HEIGHT = 5

# Button action labels (ButtonAction enum value -> name)
BUTTON_ACTION_LABELS = {
    BUTTON_ACTION_TOGGLE_PAUSE: "Toggle Pause",
    BUTTON_ACTION_TOGGLE_DIRECTION: "Toggle Direction",
    BUTTON_ACTION_TOGGLE_BOUNCE: "Toggle Bounce",
    BUTTON_ACTION_TRIGGER_MOVE: "Trigger Move",
    BUTTON_ACTION_TOGGLE_BARRIER: "Toggle Barrier",
    BUTTON_ACTION_TIMED_BARRIER: "Timed Barrier",
}

BUTTON_ACTION_CHOICES = [
    (BUTTON_ACTION_TOGGLE_BARRIER, BUTTON_ACTION_LABELS[BUTTON_ACTION_TOGGLE_BARRIER]),
    (BUTTON_ACTION_TIMED_BARRIER, BUTTON_ACTION_LABELS[BUTTON_ACTION_TIMED_BARRIER]),
    (BUTTON_ACTION_TOGGLE_PAUSE, BUTTON_ACTION_LABELS[BUTTON_ACTION_TOGGLE_PAUSE]),
    (BUTTON_ACTION_TOGGLE_DIRECTION, BUTTON_ACTION_LABELS[BUTTON_ACTION_TOGGLE_DIRECTION]),
    (BUTTON_ACTION_TOGGLE_BOUNCE, BUTTON_ACTION_LABELS[BUTTON_ACTION_TOGGLE_BOUNCE]),
    (BUTTON_ACTION_TRIGGER_MOVE, BUTTON_ACTION_LABELS[BUTTON_ACTION_TRIGGER_MOVE]),
]

# Which action values apply to barriers (for buttons / pressure plates)
BARRIER_ACTIONS = {BUTTON_ACTION_TOGGLE_BARRIER, BUTTON_ACTION_TIMED_BARRIER}
# Which apply to moving walls
MOVING_WALL_ACTIONS = {
    BUTTON_ACTION_TOGGLE_PAUSE,
    BUTTON_ACTION_TOGGLE_DIRECTION,
    BUTTON_ACTION_TOGGLE_BOUNCE,
    BUTTON_ACTION_TRIGGER_MOVE,
}


def _button_action_value(action_type) -> int:
    return int(action_type)


def _button_action_label(action_type) -> str:
    try:
        return BUTTON_ACTION_LABELS[_button_action_value(action_type)]
    except (KeyError, TypeError, ValueError):
        return str(action_type)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class EditorTool:
    """Represents a tool/object that can be placed."""
    name: str
    symbol: str
    description: str
    color: Tuple[int, int, int]
    static_object: Optional[int] = None
    conveyor_direction: Optional[int] = None
    is_agent: bool = False
    is_erase: bool = False
    is_button: bool = False
    is_barrier: bool = False
    is_pressure_plate: bool = False
    keyboard_shortcut: Optional[str] = None


def _make_tools():
    """Build the tools list (called after deps are loaded so we can use StaticObject)."""
    def _so(name, fallback):
        return getattr(StaticObject, name) if DEPENDENCIES_AVAILABLE else fallback

    return [
        EditorTool("Wall",           "W",  "Blocks movement / acts as counter.",      COLOR_GRAY,
                   static_object=_so("WALL", FALLBACK_WALL), keyboard_shortcut="w"),
        EditorTool("Pot",            "P",  "Cooks ingredients into dishes.",           COLOR_ORANGE,
                   static_object=_so("POT", FALLBACK_POT), keyboard_shortcut="p"),
        EditorTool("Plate Pile",     "B",  "Provides clean plates for serving.",      COLOR_WHITE,
                   static_object=_so("PLATE_PILE", FALLBACK_PLATE_PILE), keyboard_shortcut="b"),
        EditorTool("Delivery",       "X",  "Deliver completed dishes here.",          COLOR_GREEN,
                   static_object=_so("GOAL", FALLBACK_GOAL), keyboard_shortcut="x"),
        EditorTool("Recipe",         "R",  "Shows the current recipe to cook.",       COLOR_PURPLE,
                   static_object=_so("RECIPE_INDICATOR", FALLBACK_RECIPE), keyboard_shortcut="r"),
        EditorTool("Agent",          "A",  "Sets an agent spawn position.",           COLOR_BLUE,
                   is_agent=True, keyboard_shortcut="a"),
        EditorTool("Ingredient 0",   "0",  "Onion pile (ingredient source).",         COLOR_YELLOW,
                   static_object=(_so("INGREDIENT_PILE_BASE", FALLBACK_INGREDIENT_BASE)),
                   keyboard_shortcut="0"),
        EditorTool("Ingredient 1",   "1",  "Tomato pile (ingredient source).",        COLOR_RED,
                   static_object=(_so("INGREDIENT_PILE_BASE", FALLBACK_INGREDIENT_BASE) + 1),
                   keyboard_shortcut="1"),
        EditorTool("Ingredient 2",   "2",  "Lettuce pile (ingredient source).",       (0, 150, 0),
                   static_object=(_so("INGREDIENT_PILE_BASE", FALLBACK_INGREDIENT_BASE) + 2),
                   keyboard_shortcut="2"),
        EditorTool("Item Conv >",    ">",  "Moves items to the right.",               COLOR_CYAN,
                   conveyor_direction=2, keyboard_shortcut=">"),
        EditorTool("Item Conv <",    "<",  "Moves items to the left.",                COLOR_CYAN,
                   conveyor_direction=3, keyboard_shortcut="<"),
        EditorTool("Item Conv ^",    "^",  "Moves items upward.",                     COLOR_CYAN,
                   conveyor_direction=0, keyboard_shortcut="^"),
        EditorTool("Item Conv v",    "v",  "Moves items downward.",                   COLOR_CYAN,
                   conveyor_direction=1, keyboard_shortcut="v"),
        EditorTool("Player Conv ]",  "]",  "Moves agents to the right.",              COLOR_PURPLE,
                   conveyor_direction=2, keyboard_shortcut="]"),
        EditorTool("Player Conv [",  "[",  "Moves agents to the left.",               COLOR_PURPLE,
                   conveyor_direction=3, keyboard_shortcut="["),
        EditorTool("Player Conv {",  "{",  "Moves agents upward.",                    COLOR_PURPLE,
                   conveyor_direction=0, keyboard_shortcut="{"),
        EditorTool("Player Conv }",  "}",  "Moves agents downward.",                  COLOR_PURPLE,
                   conveyor_direction=1, keyboard_shortcut="}"),
        # --- New tools ---
        EditorTool("Button",         "Q",  "Interactive button; links to barriers/walls.", COLOR_TEAL,
                   static_object=_so("BUTTON", FALLBACK_BUTTON),
                   is_button=True, keyboard_shortcut="q"),
        EditorTool("Barrier",        "K",  "Toggleable wall tile (starts active).",   COLOR_MAROON,
                   static_object=_so("BARRIER", FALLBACK_BARRIER),
                   is_barrier=True, keyboard_shortcut="k"),
        EditorTool("Pressure Plate", "L",  "Floor plate; activates when stepped on.", COLOR_LIME,
                   static_object=_so("PRESSURE_PLATE", FALLBACK_PRESSURE_PLATE),
                   is_pressure_plate=True, keyboard_shortcut="l"),
        EditorTool("Erase",          "⌫",  "Removes objects from a tile.",           COLOR_RED,
                   is_erase=True, keyboard_shortcut="e"),
    ]


# Lazily populated after _load_jaxmarl_deps()
TOOLS: List[EditorTool] = []
SHORTCUT_TO_TOOL: Dict[str, int] = {}


def _init_tools():
    global TOOLS, SHORTCUT_TO_TOOL
    TOOLS = _make_tools()
    SHORTCUT_TO_TOOL = {
        tool.keyboard_shortcut: i
        for i, tool in enumerate(TOOLS)
        if tool.keyboard_shortcut
    }


# ---------------------------------------------------------------------------
# Button / Barrier / Pressure-plate data structures
# ---------------------------------------------------------------------------

@dataclass
class ButtonInfo:
    """Data for a single button placed in the editor."""
    y: int
    x: int
    action_type: int = DEFAULT_WIRE_ACTION  # ButtonAction enum value
    target_barrier_idxs: List[int] = field(default_factory=list)  # indices into EditorState.barriers


@dataclass
class BarrierInfo:
    """Data for a single barrier placed in the editor."""
    y: int
    x: int
    initially_active: bool = True  # starts as a solid wall


@dataclass
class PressurePlateInfo:
    """Data for a single pressure plate placed in the editor."""
    y: int
    x: int
    action_type: int = DEFAULT_WIRE_ACTION  # ButtonAction enum value
    target_barrier_idxs: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Editor state
# ---------------------------------------------------------------------------

@dataclass
class EditorState:
    """Current state of the level editor."""
    width: int = DEFAULT_GRID_WIDTH
    height: int = DEFAULT_GRID_HEIGHT
    static_objects: np.ndarray = field(
        default_factory=lambda: np.zeros((DEFAULT_GRID_HEIGHT, DEFAULT_GRID_WIDTH), dtype=int)
    )
    agent_positions: List[Tuple[int, int]] = field(default_factory=list)
    item_conveyors: Dict[Tuple[int, int], int] = field(default_factory=dict)
    player_conveyors: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # New: buttons, barriers, pressure plates
    buttons: List[ButtonInfo] = field(default_factory=list)
    barriers: List[BarrierInfo] = field(default_factory=list)
    pressure_plates: List[PressurePlateInfo] = field(default_factory=list)

    selected_tool: int = 0
    undo_stack: List[dict] = field(default_factory=list)
    redo_stack: List[dict] = field(default_factory=list)
    layout_name: str = "custom_layout"
    recipes: List[List[int]] = field(default_factory=lambda: [[0, 0, 0]])
    modified: bool = False

    # ---- wiring UI state (not saved in undo) ----
    selected_cell: Optional[Tuple[int, int]] = None  # (x, y) of highlighted cell

    def clone(self) -> dict:
        return {
            "static_objects": self.static_objects.copy(),
            "agent_positions": self.agent_positions.copy(),
            "item_conveyors": self.item_conveyors.copy(),
            "player_conveyors": self.player_conveyors.copy(),
            "buttons": [ButtonInfo(b.y, b.x, b.action_type, b.target_barrier_idxs.copy())
                        for b in self.buttons],
            "barriers": [BarrierInfo(b.y, b.x, b.initially_active) for b in self.barriers],
            "pressure_plates": [PressurePlateInfo(p.y, p.x, p.action_type, p.target_barrier_idxs.copy())
                                 for p in self.pressure_plates],
        }

    def restore(self, snapshot: dict):
        self.static_objects = snapshot["static_objects"]
        self.agent_positions = snapshot["agent_positions"]
        self.item_conveyors = snapshot["item_conveyors"]
        self.player_conveyors = snapshot["player_conveyors"]
        self.buttons = snapshot.get("buttons", [])
        self.barriers = snapshot.get("barriers", [])
        self.pressure_plates = snapshot.get("pressure_plates", [])

    def save_undo(self):
        self.undo_stack.append(self.clone())
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.clone())
            self.restore(self.undo_stack.pop())
            self.modified = True

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.clone())
            self.restore(self.redo_stack.pop())
            self.modified = True

    def resize(self, new_width: int, new_height: int):
        self.save_undo()
        new_grid = np.zeros((new_height, new_width), dtype=int)
        copy_h = min(self.height, new_height)
        copy_w = min(self.width, new_width)
        new_grid[:copy_h, :copy_w] = self.static_objects[:copy_h, :copy_w]
        self.agent_positions = [(x, y) for x, y in self.agent_positions if x < new_width and y < new_height]
        self.item_conveyors = {(y, x): d for (y, x), d in self.item_conveyors.items() if x < new_width and y < new_height}
        self.player_conveyors = {(y, x): d for (y, x), d in self.player_conveyors.items() if x < new_width and y < new_height}
        self.buttons = [b for b in self.buttons if b.x < new_width and b.y < new_height]
        self.barriers = [b for b in self.barriers if b.x < new_width and b.y < new_height]
        self.pressure_plates = [p for p in self.pressure_plates if p.x < new_width and p.y < new_height]
        # Remap barrier indices in buttons and pressure plates
        self._remap_barrier_references()
        self.static_objects = new_grid
        self.width = new_width
        self.height = new_height
        self.modified = True

    def _remap_barrier_references(self):
        """Rebuild barrier index references after barriers list may have changed."""
        # Build set of valid barrier indices
        valid = set(range(len(self.barriers)))
        for b in self.buttons:
            b.target_barrier_idxs = [i for i in b.target_barrier_idxs if i in valid]
        for p in self.pressure_plates:
            p.target_barrier_idxs = [i for i in p.target_barrier_idxs if i in valid]

    def clear(self):
        self.save_undo()
        self.static_objects = np.zeros((self.height, self.width), dtype=int)
        self.agent_positions.clear()
        self.item_conveyors.clear()
        self.player_conveyors.clear()
        self.buttons.clear()
        self.barriers.clear()
        self.pressure_plates.clear()
        self.modified = True

    # ---- Lookup helpers ----

    def barrier_at(self, x: int, y: int) -> Optional[int]:
        """Return the index of a barrier at (x, y), or None."""
        for i, b in enumerate(self.barriers):
            if b.x == x and b.y == y:
                return i
        return None

    def button_at(self, x: int, y: int) -> Optional[int]:
        for i, b in enumerate(self.buttons):
            if b.x == x and b.y == y:
                return i
        return None

    def pressure_plate_at(self, x: int, y: int) -> Optional[int]:
        for i, p in enumerate(self.pressure_plates):
            if p.x == x and p.y == y:
                return i
        return None

    # ---- Export helpers ----

    def to_layout(self):
        """Convert editor state to a Layout object."""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Cannot create Layout without JaxMARL dependencies")

        item_conveyor_info = [(y, x, d) for (y, x), d in self.item_conveyors.items()]
        player_conveyor_info = [(y, x, d) for (y, x), d in self.player_conveyors.items()]

        # Barrier info: list of (y, x, initially_active)
        barrier_info = [(b.y, b.x, b.initially_active) for b in self.barriers]

        # Button info: list of (y, x, target_idxs_tuple, action_type)
        button_info = []
        for btn in self.buttons:
            targets = tuple(btn.target_barrier_idxs) if btn.target_barrier_idxs else (0,)
            button_info.append((btn.y, btn.x, targets, btn.action_type))

        # Pressure plate info: list of (y, x, barrier_target_list, action_type)
        pressure_plate_info = []
        for pp in self.pressure_plates:
            targets = list(pp.target_barrier_idxs)
            pressure_plate_info.append((pp.y, pp.x, targets, pp.action_type))

        num_ingredients = 1
        if DEPENDENCIES_AVAILABLE:
            ingredient_base = StaticObject.INGREDIENT_PILE_BASE
            item_conveyor_val = StaticObject.ITEM_CONVEYOR
        else:
            ingredient_base = FALLBACK_INGREDIENT_BASE
            item_conveyor_val = FALLBACK_ITEM_CONVEYOR
        for obj in self.static_objects.flat:
            if ingredient_base <= obj < item_conveyor_val:
                ingredient_idx = obj - ingredient_base
                num_ingredients = max(num_ingredients, ingredient_idx + 1)

        return Layout(
            agent_positions=self.agent_positions,
            static_objects=self.static_objects.copy(),
            num_ingredients=num_ingredients,
            possible_recipes=self.recipes if self.recipes else None,
            item_conveyor_info=item_conveyor_info,
            player_conveyor_info=player_conveyor_info,
            barrier_info=barrier_info,
            button_info=button_info,
            pressure_plate_info=pressure_plate_info,
        )


# ---------------------------------------------------------------------------
# Main editor application
# ---------------------------------------------------------------------------

class LevelEditor:
    """Main level editor application."""

    def __init__(self):
        pygame.init()

        self.tile_size = TILE_SIZE
        self.grid_width = DEFAULT_GRID_WIDTH * self.tile_size
        self.grid_height = DEFAULT_GRID_HEIGHT * self.tile_size
        self.window_width = TOOLBAR_WIDTH + self.grid_width * 2
        self.window_height = TOP_MENU_HEIGHT + self.grid_height

        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Overcooked V3 Level Editor")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 32)

        self.state = EditorState()
        self.running = True
        self.hover_pos = None

        # Dialog flags
        self.show_export_dialog = False
        self.export_text = ""
        self.show_load_dialog = False
        self.selected_layout_name = None
        self.validation_messages = []

        # Resize dialog
        self.show_resize_dialog = False
        self.resize_width_input = str(DEFAULT_GRID_WIDTH)
        self.resize_height_input = str(DEFAULT_GRID_HEIGHT)
        self.resize_input_mode = "width"
        self.resize_error = ""

        # Paint drag
        self.painting = False
        self.paint_button = None
        self.last_paint_cell = None

        # Wiring panel state
        self.wiring_cell: Optional[Tuple[int, int]] = None  # (x, y) being wired

        # Hit-test rects populated by _draw_wiring_panel / _draw_barrier_panel each frame
        # so that _handle_info_panel_click always uses the exact rendered positions.
        self._wiring_action_rects: List[Tuple[pygame.Rect, int]] = []  # rect -> ButtonAction value
        self._wiring_barrier_rects: List[pygame.Rect] = []  # index → barrier checkbox
        self._barrier_toggle_rect: Optional[pygame.Rect] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)
        pygame.quit()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.button, event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.painting = False
                self.paint_button = None
                self.last_paint_cell = None
            elif event.type == pygame.MOUSEMOTION:
                self.hover_pos = event.pos
                if self.painting:
                    self._handle_paint_drag(event.pos)
            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event)
            elif event.type == pygame.VIDEORESIZE:
                self._set_window_size(event.w, event.h)

    def _set_window_size(self, width: int, height: int):
        self.window_width = width
        self.window_height = height
        self._recalc_tile_size()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

    def _recalc_tile_size(self):
        available = self.window_width - TOOLBAR_WIDTH
        half = available // 2
        self.tile_size = max(16, half // self.state.width)
        self.grid_width = self.state.width * self.tile_size
        self.grid_height = self.state.height * self.tile_size

    # --- Click routing ---

    def handle_mouse_click(self, button: int, pos: Tuple[int, int]):
        mx, my = pos

        # Toolbar
        if mx < TOOLBAR_WIDTH:
            self.handle_toolbar_click(mx, my)
            return

        # Top menu
        if my < TOP_MENU_HEIGHT:
            self.handle_menu_click(mx, my)
            return

        # Info panel (right half)
        grid_area_width = self.state.width * self.tile_size
        if mx >= TOOLBAR_WIDTH + grid_area_width:
            self._handle_info_panel_click(mx, my, button)
            return

        # Grid
        grid_x = mx - TOOLBAR_WIDTH
        grid_y = my - TOP_MENU_HEIGHT
        if 0 <= grid_x < grid_area_width and 0 <= grid_y < self.state.height * self.tile_size:
            cell_x = grid_x // self.tile_size
            cell_y = grid_y // self.tile_size
            if button == 1:
                self.place_object(cell_x, cell_y)
                self.painting = True
                self.paint_button = 1
                self.last_paint_cell = (cell_x, cell_y)
                # Open wiring/config panel for interactive elements
                so = self.state.static_objects[cell_y, cell_x]
                if self._is_wireable(so, cell_x, cell_y):
                    self.wiring_cell = (cell_x, cell_y)
            elif button == 3:
                self.erase_object(cell_x, cell_y)
                self.painting = True
                self.paint_button = 3
                self.last_paint_cell = (cell_x, cell_y)

    def _is_wireable(self, static_obj: int, x: int, y: int) -> bool:
        """True if this cell has a button, pressure plate, or barrier (all have config panels)."""
        if self.state.button_at(x, y) is not None:
            return True
        if self.state.pressure_plate_at(x, y) is not None:
            return True
        if self.state.barrier_at(x, y) is not None:
            return True
        return False

    def _handle_info_panel_click(self, mx: int, my: int, button: int):
        """Handle clicks in the right info / wiring panel.

        All hit-test rectangles are captured by the draw methods each frame,
        so the click handler never has to recompute positions independently.
        """
        if button != 1:
            return
        if self.wiring_cell is None:
            return

        wx, wy = self.wiring_cell
        btn_idx = self.state.button_at(wx, wy)
        pp_idx  = self.state.pressure_plate_at(wx, wy)
        bar_idx = self.state.barrier_at(wx, wy)

        # --- Wiring panel (button or pressure plate selected) ---
        if btn_idx is not None or pp_idx is not None:
            obj = self.state.buttons[btn_idx] if btn_idx is not None else self.state.pressure_plates[pp_idx]

            # Action type rows
            for rect, action_type in self._wiring_action_rects:
                if rect.collidepoint(mx, my):
                    obj.action_type = action_type
                    return

            # Barrier checkbox rows
            for i, rect in enumerate(self._wiring_barrier_rects):
                if rect.collidepoint(mx, my):
                    if i in obj.target_barrier_idxs:
                        obj.target_barrier_idxs.remove(i)
                    else:
                        obj.target_barrier_idxs.append(i)
                    return

        # --- Barrier panel (barrier tile selected) ---
        if bar_idx is not None and self._barrier_toggle_rect is not None:
            if self._barrier_toggle_rect.collidepoint(mx, my):
                self.state.barriers[bar_idx].initially_active = not self.state.barriers[bar_idx].initially_active

    def _handle_paint_drag(self, pos: Tuple[int, int]):
        mx, my = pos
        grid_x = mx - TOOLBAR_WIDTH
        grid_y = my - TOP_MENU_HEIGHT
        if 0 <= grid_x < self.state.width * self.tile_size and 0 <= grid_y < self.state.height * self.tile_size:
            cell_x = grid_x // self.tile_size
            cell_y = grid_y // self.tile_size
            if (cell_x, cell_y) == self.last_paint_cell:
                return
            self.last_paint_cell = (cell_x, cell_y)
            if self.paint_button == 1:
                tool = TOOLS[self.state.selected_tool]
                if tool.is_agent:
                    return
                self.place_object(cell_x, cell_y, save_undo=False)
            elif self.paint_button == 3:
                self.erase_object(cell_x, cell_y, save_undo=False)
        else:
            self.painting = False
            self.paint_button = None
            self.last_paint_cell = None

    def handle_toolbar_click(self, mx: int, my: int):
        toolbar_y = my - TOP_MENU_HEIGHT
        if toolbar_y < 0:
            return
        tool_height = 35
        tool_idx = toolbar_y // tool_height
        if 0 <= tool_idx < len(TOOLS):
            self.state.selected_tool = tool_idx

    def handle_menu_click(self, mx: int, my: int):
        menu_items = ["New", "Load", "Export", "Resize", "Test", "Quit"]
        item_width = 80
        item_idx = (mx - TOOLBAR_WIDTH) // item_width
        if 0 <= item_idx < len(menu_items):
            action = menu_items[item_idx]
            if action == "New":
                self.new_layout()
            elif action == "Load":
                self.show_load_dialog = True
            elif action == "Export":
                self.export_layout()
            elif action == "Resize":
                self.show_resize_dialog = True
                self.resize_error = ""
                self.resize_width_input = str(self.state.width)
                self.resize_height_input = str(self.state.height)
            elif action == "Test":
                self.test_play()
            elif action == "Quit":
                self.running = False

    def handle_keypress(self, event):
        key = pygame.key.name(event.key)
        mods = pygame.key.get_mods()

        # Resize dialog
        if self.show_resize_dialog:
            if key == "backspace":
                if self.resize_input_mode == "width":
                    self.resize_width_input = self.resize_width_input[:-1]
                else:
                    self.resize_height_input = self.resize_height_input[:-1]
            elif key == "tab":
                self.resize_input_mode = "height" if self.resize_input_mode == "width" else "width"
            elif key == "return":
                try:
                    nw = int(self.resize_width_input)
                    nh = int(self.resize_height_input)
                    if MIN_GRID_WIDTH <= nw <= MAX_GRID_WIDTH and MIN_GRID_HEIGHT <= nh <= MAX_GRID_HEIGHT:
                        self.state.resize(nw, nh)
                        self.show_resize_dialog = False
                        self._recalc_tile_size()
                except ValueError:
                    pass
            elif key in "0123456789":
                if self.resize_input_mode == "width":
                    if len(self.resize_width_input) < 2:
                        self.resize_width_input += key
                else:
                    if len(self.resize_height_input) < 2:
                        self.resize_height_input += key
            elif key == "escape":
                self.show_resize_dialog = False
            return

        # Dismiss wiring panel with Escape
        if key == "escape" and self.wiring_cell is not None:
            self.wiring_cell = None
            return

        char = event.unicode
        if char in SHORTCUT_TO_TOOL:
            self.state.selected_tool = SHORTCUT_TO_TOOL[char]
            return
        if key in SHORTCUT_TO_TOOL:
            self.state.selected_tool = SHORTCUT_TO_TOOL[key]
            return

        if mods & pygame.KMOD_CTRL:
            if key == "z":
                self.state.undo()
            elif key == "y":
                self.state.redo()
            elif key == "n":
                self.new_layout()
            elif key == "e":
                self.export_layout()
            elif key == "t":
                self.test_play()

    # ------------------------------------------------------------------
    # Place / Erase
    # ------------------------------------------------------------------

    def place_object(self, x: int, y: int, save_undo: bool = True):
        if save_undo:
            self.state.save_undo()

        tool = TOOLS[self.state.selected_tool]
        self.erase_object(x, y, save_undo=False)

        if tool.is_agent:
            self.state.agent_positions.append((x, y))

        elif tool.is_erase:
            pass  # already erased

        elif tool.is_button:
            if DEPENDENCIES_AVAILABLE:
                self.state.static_objects[y, x] = StaticObject.BUTTON
            else:
                self.state.static_objects[y, x] = FALLBACK_BUTTON
            self.state.buttons.append(ButtonInfo(y=y, x=x))
            self.wiring_cell = (x, y)

        elif tool.is_barrier:
            if DEPENDENCIES_AVAILABLE:
                self.state.static_objects[y, x] = StaticObject.BARRIER
            else:
                self.state.static_objects[y, x] = FALLBACK_BARRIER
            self.state.barriers.append(BarrierInfo(y=y, x=x, initially_active=True))

        elif tool.is_pressure_plate:
            if DEPENDENCIES_AVAILABLE:
                self.state.static_objects[y, x] = StaticObject.PRESSURE_PLATE
            else:
                self.state.static_objects[y, x] = FALLBACK_PRESSURE_PLATE
            self.state.pressure_plates.append(PressurePlateInfo(y=y, x=x))
            self.wiring_cell = (x, y)

        elif tool.conveyor_direction is not None:
            if "Player" in tool.name:
                self.state.player_conveyors[(y, x)] = tool.conveyor_direction
                self.state.static_objects[y, x] = (
                    StaticObject.PLAYER_CONVEYOR if DEPENDENCIES_AVAILABLE else FALLBACK_PLAYER_CONVEYOR
                )
            else:
                self.state.item_conveyors[(y, x)] = tool.conveyor_direction
                self.state.static_objects[y, x] = (
                    StaticObject.ITEM_CONVEYOR if DEPENDENCIES_AVAILABLE else FALLBACK_ITEM_CONVEYOR
                )

        elif tool.static_object is not None:
            self.state.static_objects[y, x] = tool.static_object

        self.state.modified = True

    def erase_object(self, x: int, y: int, save_undo: bool = True):
        if save_undo:
            self.state.save_undo()

        self.state.agent_positions = [(ax, ay) for ax, ay in self.state.agent_positions if not (ax == x and ay == y)]

        if (y, x) in self.state.item_conveyors:
            del self.state.item_conveyors[(y, x)]
        if (y, x) in self.state.player_conveyors:
            del self.state.player_conveyors[(y, x)]

        # Remove barrier and fix up references
        bar_idx = self.state.barrier_at(x, y)
        if bar_idx is not None:
            self.state.barriers.pop(bar_idx)
            # Remap remaining references (shift indices > bar_idx down by 1)
            for btn in self.state.buttons:
                btn.target_barrier_idxs = [
                    i if i < bar_idx else i - 1
                    for i in btn.target_barrier_idxs if i != bar_idx
                ]
            for pp in self.state.pressure_plates:
                pp.target_barrier_idxs = [
                    i if i < bar_idx else i - 1
                    for i in pp.target_barrier_idxs if i != bar_idx
                ]

        # Remove button
        btn_idx = self.state.button_at(x, y)
        if btn_idx is not None:
            self.state.buttons.pop(btn_idx)
            if self.wiring_cell == (x, y):
                self.wiring_cell = None

        # Remove pressure plate
        pp_idx = self.state.pressure_plate_at(x, y)
        if pp_idx is not None:
            self.state.pressure_plates.pop(pp_idx)
            if self.wiring_cell == (x, y):
                self.wiring_cell = None

        self.state.static_objects[y, x] = 0

        if save_undo:
            self.state.modified = True

    # ------------------------------------------------------------------
    # Layout operations
    # ------------------------------------------------------------------

    def new_layout(self):
        self.state = EditorState()
        self.validation_messages = []
        self.show_load_dialog = False
        self.show_export_dialog = False
        self.wiring_cell = None

    def export_layout(self):
        try:
            layout_str = self._layout_string_from_state()
            barrier_code = self._barrier_code()
            button_code = self._button_code()
            pressure_plate_code = self._pressure_plate_code()
            barrier_config_code = self._barrier_config_list()
            button_config_code = self._button_config_list()
            pressure_plate_config_code = self._pressure_plate_config_list()

            code = f'''# Add to jaxmarl/environments/overcooked_v3/layouts.py

{self.state.layout_name} = """
{layout_str.strip()}
"""

{barrier_code}

overcooked_v3_layouts["{self.state.layout_name}"] = Layout.from_string(
    {self.state.layout_name},
    possible_recipes={self.state.recipes},
    swap_agents=False,
    barrier_config={barrier_config_code},
    button_config={button_config_code},
    pressure_plate_config={pressure_plate_config_code},
)
'''
            export_dir = Path(__file__).resolve().parents[2] / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"{self.state.layout_name}.txt"
            export_path.write_text(code, encoding="utf-8")

            self.export_text = str(export_path)
            self.show_export_dialog = True
            print("\n" + "=" * 60)
            print("EXPORTED LAYOUT CODE:")
            print("=" * 60)
            print(code)
            print(f"Saved to: {export_path}")
            print("=" * 60)
        except Exception as e:
            print(f"Error exporting layout: {e}")
            self.validation_messages = [f"Export error: {e}"]

    def _ordered_barriers(self) -> List[Tuple[int, BarrierInfo]]:
        return sorted(enumerate(self.state.barriers), key=lambda item: (item[1].y, item[1].x))

    def _ordered_buttons(self) -> List[Tuple[int, ButtonInfo]]:
        return sorted(enumerate(self.state.buttons), key=lambda item: (item[1].y, item[1].x))

    def _ordered_pressure_plates(self) -> List[Tuple[int, PressurePlateInfo]]:
        return sorted(enumerate(self.state.pressure_plates), key=lambda item: (item[1].y, item[1].x))

    def _barrier_export_index_map(self) -> Dict[int, int]:
        return {
            editor_idx: export_idx
            for export_idx, (editor_idx, _) in enumerate(self._ordered_barriers())
        }

    def _remap_barrier_targets_for_export(
        self,
        targets: List[int],
        barrier_index_map: Dict[int, int],
    ) -> Tuple[int, ...]:
        remapped = []
        for target_idx in targets:
            try:
                target_idx = int(target_idx)
            except (TypeError, ValueError):
                remapped.append(target_idx)
                continue
            remapped.append(barrier_index_map.get(target_idx, target_idx))
        return tuple(remapped) if remapped else (0,)

    def _barrier_config(self) -> List[bool]:
        return [bool(barrier.initially_active) for _, barrier in self._ordered_barriers()]

    def _button_config(self) -> List[Tuple[Tuple[int, ...], int]]:
        barrier_index_map = self._barrier_export_index_map()
        items = []
        for _, btn in self._ordered_buttons():
            targets = self._remap_barrier_targets_for_export(
                btn.target_barrier_idxs,
                barrier_index_map,
            )
            items.append((targets, _button_action_value(btn.action_type)))
        return items

    def _pressure_plate_config(self) -> List[Tuple[Tuple[int, ...], int]]:
        barrier_index_map = self._barrier_export_index_map()
        items = []
        for _, pp in self._ordered_pressure_plates():
            targets = self._remap_barrier_targets_for_export(
                pp.target_barrier_idxs,
                barrier_index_map,
            )
            items.append((targets, _button_action_value(pp.action_type)))
        return items

    def _barrier_config_list(self) -> str:
        return repr(self._barrier_config())

    def _button_config_list(self) -> str:
        return repr(self._button_config())

    def _pressure_plate_config_list(self) -> str:
        return repr(self._pressure_plate_config())

    def _barrier_code(self) -> str:
        if not self.state.barriers:
            return "# No barriers defined"
        lines = ["# Barrier definitions: (export_idx, y, x, initially_active)"]
        for i, (_, b) in enumerate(self._ordered_barriers()):
            lines.append(f"# Barrier {i}: ({b.y}, {b.x}, {b.initially_active})")
        return "\n".join(lines)

    def _button_code(self) -> str:
        if not self.state.buttons:
            return ""
        lines = ["# Button definitions: (y, x, target_barrier_indices, action_type)"]
        for i, ((_, btn), (targets, action_type)) in enumerate(zip(self._ordered_buttons(), self._button_config())):
            action_name = _button_action_label(action_type)
            lines.append(f"# Button {i}: ({btn.y}, {btn.x}, {targets}, {action_type})  # {action_name}")
        return "\n".join(lines)

    def _pressure_plate_code(self) -> str:
        if not self.state.pressure_plates:
            return ""
        lines = ["# Pressure plate definitions: (y, x, target_barrier_indices, action_type)"]
        for i, ((_, pp), (targets, action_type)) in enumerate(zip(self._ordered_pressure_plates(), self._pressure_plate_config())):
            action_name = _button_action_label(action_type)
            lines.append(f"# Pressure Plate {i}: ({pp.y}, {pp.x}, {targets}, {action_type})  # {action_name}")
        return "\n".join(lines)

    def _layout_string_from_state(self) -> str:
        height, width = self.state.height, self.state.width
        grid = [[" " for _ in range(width)] for _ in range(height)]

        if DEPENDENCIES_AVAILABLE and StaticObject is not None:
            ingredient_base = StaticObject.INGREDIENT_PILE_BASE
            item_conveyor_val = StaticObject.ITEM_CONVEYOR
            static_to_symbol = {
                StaticObject.WALL: "W",
                StaticObject.GOAL: "X",
                StaticObject.PLATE_PILE: "B",
                StaticObject.POT: "P",
                StaticObject.RECIPE_INDICATOR: "R",
                StaticObject.BUTTON: "!",
                StaticObject.BARRIER: "#",
                StaticObject.PRESSURE_PLATE: "_",
            }
        else:
            ingredient_base = FALLBACK_INGREDIENT_BASE
            item_conveyor_val = FALLBACK_ITEM_CONVEYOR
            static_to_symbol = {
                FALLBACK_WALL: "W",
                FALLBACK_GOAL: "X",
                FALLBACK_PLATE_PILE: "B",
                FALLBACK_POT: "P",
                FALLBACK_RECIPE: "R",
                FALLBACK_BUTTON: "!",
                FALLBACK_BARRIER: "#",
                FALLBACK_PRESSURE_PLATE: "_",
            }

        item_symbols = {2: ">", 3: "<", 0: "^", 1: "v"}
        player_symbols = {2: "]", 3: "[", 0: "{", 1: "}"}

        for y in range(height):
            for x in range(width):
                if (y, x) in self.state.item_conveyors:
                    grid[y][x] = item_symbols.get(self.state.item_conveyors[(y, x)], ">")
                    continue
                if (y, x) in self.state.player_conveyors:
                    grid[y][x] = player_symbols.get(self.state.player_conveyors[(y, x)], "]")
                    continue
                obj = self.state.static_objects[y, x]
                if obj in static_to_symbol:
                    grid[y][x] = static_to_symbol[obj]
                elif ingredient_base <= obj < item_conveyor_val:
                    grid[y][x] = str(obj - ingredient_base)

        for agent_x, agent_y in self.state.agent_positions:
            if 0 <= agent_x < width and 0 <= agent_y < height:
                grid[agent_y][agent_x] = "A"

        return "\n" + "\n".join("".join(row) for row in grid) + "\n"

    def test_play(self):
        if not DEPENDENCIES_AVAILABLE:
            print("Cannot test play: JaxMARL dependencies not available")
            return
        try:
            layout = self.state.to_layout()
            is_valid, messages = layout.validate_playable()
            if not is_valid:
                self.validation_messages = ["Cannot test - layout has errors:"] + messages
                return

            import tempfile, subprocess
            layout_str = self._layout_string_from_state()
            recipes_str = str(self.state.recipes)
            barrier_config_str = self._barrier_config_list()
            button_config_str = self._button_config_list()
            pp_config_str = self._pressure_plate_config_list()

            test_script = f'''#!/usr/bin/env python3
import jax
import pygame
import numpy as np
from jaxmarl import make
from jaxmarl.environments.overcooked_v3.layouts import Layout
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

layout_str = """{layout_str}"""
layout = Layout.from_string(
    layout_str,
    possible_recipes={recipes_str},
    barrier_config={barrier_config_str},
    button_config={button_config_str},
    pressure_plate_config={pp_config_str},
)

env = make("overcooked_v3")
env.layout = layout
viz = OvercookedV3Visualizer(env, tile_size=48)
pygame.init()
screen = pygame.display.set_mode((env.width * 48, env.height * 48))
pygame.display.set_caption("Test Play - Q to quit")
clock = pygame.time.Clock()
key = jax.random.PRNGKey(42)
key, sk = jax.random.split(key)
obs, state = env.reset(sk)
AGENT0 = {{pygame.K_w:3,pygame.K_s:1,pygame.K_a:2,pygame.K_d:0,pygame.K_SPACE:5}}
AGENT1 = {{pygame.K_UP:3,pygame.K_DOWN:1,pygame.K_LEFT:2,pygame.K_RIGHT:0,pygame.K_RETURN:5}}
total = 0
running = True
while running:
    a0, a1 = 4, 4
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_q, pygame.K_ESCAPE): running = False
            if e.key == pygame.K_r:
                key, sk = jax.random.split(key); obs, state = env.reset(sk); total = 0
    keys = pygame.key.get_pressed()
    for k, v in AGENT0.items():
        if keys[k]: a0 = v; break
    for k, v in AGENT1.items():
        if keys[k]: a1 = v; break
    key, sk = jax.random.split(key)
    obs, state, rewards, dones, info = env.step(sk, state, {{"agent_0":a0,"agent_1":a1}})
    total += rewards["agent_0"]
    img = np.array(viz.render_state(state))
    surf = pygame.surfarray.make_surface(img.swapaxes(0,1))
    screen.blit(surf, (0,0))
    f = pygame.font.Font(None, 24)
    screen.blit(f.render(f"Score: {{total:.0f}}  Q=quit  R=reset", True, (255,255,255)), (5,5))
    pygame.display.flip()
    clock.tick(10)
pygame.quit()
'''
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_script)
                tmp = f.name
            subprocess.run(["python3", tmp])
            import os; os.unlink(tmp)
        except Exception as e:
            print(f"Error launching test play: {e}")
            import traceback; traceback.print_exc()
            self.validation_messages = [f"Test play error: {e}"]

    def load_layout(self, layout_name: str):
        if not DEPENDENCIES_AVAILABLE:
            return
        try:
            layout = overcooked_v3_layouts[layout_name]
            ns = EditorState()
            ns.width = layout.width
            ns.height = layout.height
            ns.static_objects = layout.static_objects.copy()
            ns.agent_positions = layout.agent_positions.copy()
            ns.recipes = layout.possible_recipes.copy() if layout.possible_recipes else [[0, 0, 0]]
            ns.layout_name = layout_name + "_modified"
            for y, x, d in layout.item_conveyor_info:
                ns.item_conveyors[(y, x)] = d
            for y, x, d in layout.player_conveyor_info:
                ns.player_conveyors[(y, x)] = d
            # Load barriers
            for i, (y, x, active) in enumerate(getattr(layout, "barrier_info", [])):
                ns.barriers.append(BarrierInfo(y=y, x=x, initially_active=active))
            # Load buttons
            for y, x, targets, action_type in getattr(layout, "button_info", []):
                ns.buttons.append(ButtonInfo(y=y, x=x, action_type=action_type,
                                              target_barrier_idxs=list(targets)))
            # Load pressure plates
            for y, x, targets, action_type in getattr(layout, "pressure_plate_info", []):
                ns.pressure_plates.append(PressurePlateInfo(y=y, x=x, action_type=action_type,
                                                              target_barrier_idxs=list(targets)))
            self.state = ns
            self.validation_messages = []
            self.wiring_cell = None
            self._recalc_tile_size()
            print(f"✓ Loaded layout: {layout_name}")
        except Exception as e:
            print(f"Error loading layout {layout_name}: {e}")
            import traceback; traceback.print_exc()
            self.validation_messages = [f"Load error: {e}"]

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self):
        self.screen.fill(COLOR_BLACK)
        self.draw_toolbar()
        self.draw_grid()
        self.draw_info_panel()
        self.draw_menu_bar()

        if self.show_export_dialog:
            self.draw_export_dialog()
        elif self.show_load_dialog:
            self.draw_load_dialog()
        elif self.show_resize_dialog:
            self.draw_resize_dialog()

        pygame.display.flip()

    def draw_menu_bar(self):
        pygame.draw.rect(self.screen, COLOR_DARK_GRAY, (0, 0, self.window_width, TOP_MENU_HEIGHT))
        menu_items = ["New", "Load", "Export", "Resize", "Test", "Quit"]
        item_width = 80
        x = TOOLBAR_WIDTH
        for item in menu_items:
            rect = pygame.Rect(x, 5, item_width - 10, TOP_MENU_HEIGHT - 10)
            pygame.draw.rect(self.screen, COLOR_GRAY, rect)
            pygame.draw.rect(self.screen, COLOR_WHITE, rect, 2)
            text = self.small_font.render(item, True, COLOR_WHITE)
            self.screen.blit(text, text.get_rect(center=rect.center))
            x += item_width

    def draw_toolbar(self):
        pygame.draw.rect(self.screen, COLOR_DARK_GRAY, (0, TOP_MENU_HEIGHT, TOOLBAR_WIDTH, self.window_height))
        y = TOP_MENU_HEIGHT + 5
        tool_height = 35
        for i, tool in enumerate(TOOLS):
            bg = COLOR_BLUE if i == self.state.selected_tool else COLOR_GRAY
            pygame.draw.rect(self.screen, bg, (5, y, TOOLBAR_WIDTH - 10, tool_height - 2))
            icon_rect = pygame.Rect(10, y + 5, 25, 25)
            pygame.draw.rect(self.screen, COLOR_WHITE, icon_rect)
            pygame.draw.rect(self.screen, COLOR_BLACK, icon_rect, 1)
            self._draw_tool_icon(tool, icon_rect)
            text = self.small_font.render(f"{tool.symbol} - {tool.name}", True, COLOR_WHITE)
            self.screen.blit(text, (40, y + 8))
            if tool.keyboard_shortcut:
                sc = self.small_font.render(f"[{tool.keyboard_shortcut}]", True, COLOR_LIGHT_GRAY)
                self.screen.blit(sc, (TOOLBAR_WIDTH - 40, y + 8))
            y += tool_height

    def _draw_tool_icon(self, tool: EditorTool, rect: pygame.Rect):
        if tool.is_erase:
            pygame.draw.rect(self.screen, COLOR_WHITE, rect)
            pygame.draw.line(self.screen, COLOR_RED, rect.topleft, rect.bottomright, 3)
            pygame.draw.line(self.screen, COLOR_RED, rect.topright, rect.bottomleft, 3)
            return
        if tool.is_agent:
            self._draw_agent(rect, 0); return
        if tool.is_button:
            self._draw_button(rect); return
        if tool.is_barrier:
            self._draw_barrier(rect, initially_active=True); return
        if tool.is_pressure_plate:
            self._draw_pressure_plate(rect); return
        if tool.conveyor_direction is not None:
            if "Player" in tool.name:
                self._draw_player_conveyor(rect, tool.conveyor_direction)
            else:
                self._draw_item_conveyor(rect, tool.conveyor_direction)
            return
        ingredient_idx = self._get_tool_ingredient_index(tool)
        if ingredient_idx is not None:
            self._draw_ingredient_pile(rect, ingredient_idx); return
        if tool.static_object is None:
            self._draw_generic(rect, 0); return
        self._dispatch_static_draw(tool.static_object, rect)

    def _dispatch_static_draw(self, obj: int, rect: pygame.Rect):
        if DEPENDENCIES_AVAILABLE:
            so = StaticObject
            if obj == so.WALL:             self._draw_wall(rect)
            elif obj == so.POT:            self._draw_pot(rect)
            elif obj == so.GOAL:           self._draw_goal(rect)
            elif obj == so.PLATE_PILE:     self._draw_plate_pile(rect)
            elif obj == so.RECIPE_INDICATOR: self._draw_recipe_indicator(rect)
            elif obj == so.BUTTON:         self._draw_button(rect)
            elif obj == so.BARRIER:        self._draw_barrier(rect)
            elif obj == so.PRESSURE_PLATE: self._draw_pressure_plate(rect)
            elif so.is_ingredient_pile(obj):
                self._draw_ingredient_pile(rect, obj - so.INGREDIENT_PILE_BASE)
            elif obj == so.ITEM_CONVEYOR:   self._draw_item_conveyor(rect, 0)
            elif obj == so.PLAYER_CONVEYOR: self._draw_player_conveyor(rect, 0)
            else: self._draw_generic(rect, obj)
        else:
            if obj == FALLBACK_WALL:           self._draw_wall(rect)
            elif obj == FALLBACK_POT:          self._draw_pot(rect)
            elif obj == FALLBACK_GOAL:         self._draw_goal(rect)
            elif obj == FALLBACK_PLATE_PILE:   self._draw_plate_pile(rect)
            elif obj == FALLBACK_RECIPE:       self._draw_recipe_indicator(rect)
            elif obj == FALLBACK_BUTTON:       self._draw_button(rect)
            elif obj == FALLBACK_BARRIER:      self._draw_barrier(rect)
            elif obj == FALLBACK_PRESSURE_PLATE: self._draw_pressure_plate(rect)
            elif FALLBACK_INGREDIENT_BASE <= obj < FALLBACK_ITEM_CONVEYOR:
                self._draw_ingredient_pile(rect, obj - FALLBACK_INGREDIENT_BASE)
            elif obj == FALLBACK_ITEM_CONVEYOR:   self._draw_item_conveyor(rect, 0)
            elif obj == FALLBACK_PLAYER_CONVEYOR:  self._draw_player_conveyor(rect, 0)
            else: self._draw_generic(rect, obj)

    def _get_tool_ingredient_index(self, tool: EditorTool) -> Optional[int]:
        if "Ingredient" in tool.name:
            try: return int(tool.name.split()[-1])
            except ValueError: return None
        if tool.static_object is None:
            return None
        if DEPENDENCIES_AVAILABLE and StaticObject is not None:
            if StaticObject.is_ingredient_pile(tool.static_object):
                return tool.static_object - StaticObject.INGREDIENT_PILE_BASE
        elif FALLBACK_INGREDIENT_BASE <= tool.static_object < FALLBACK_ITEM_CONVEYOR:
            return tool.static_object - FALLBACK_INGREDIENT_BASE
        return None

    # ---- Grid drawing ----

    def draw_grid(self):
        gx = TOOLBAR_WIDTH
        gy = TOP_MENU_HEIGHT
        grid_rect = pygame.Rect(gx, gy, self.state.width * self.tile_size, self.state.height * self.tile_size)
        pygame.draw.rect(self.screen, COLOR_WHITE, grid_rect)

        for y in range(self.state.height):
            for x in range(self.state.width):
                cell_rect = pygame.Rect(gx + x * self.tile_size, gy + y * self.tile_size,
                                        self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, COLOR_WHITE, cell_rect)
                pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY, cell_rect, 1)
                self.draw_cell_object(x, y, cell_rect)

        # Hover highlight
        if self.hover_pos:
            mx, my = self.hover_pos
            hx = mx - gx
            hy = my - gy
            if 0 <= hx < self.state.width * self.tile_size and 0 <= hy < self.state.height * self.tile_size:
                cx = hx // self.tile_size
                cy = hy // self.tile_size
                hover_rect = pygame.Rect(gx + cx * self.tile_size, gy + cy * self.tile_size,
                                         self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, COLOR_YELLOW, hover_rect, 3)

        # Wiring cell highlight
        if self.wiring_cell is not None:
            wx, wy = self.wiring_cell
            wiring_rect = pygame.Rect(gx + wx * self.tile_size, gy + wy * self.tile_size,
                                      self.tile_size, self.tile_size)
            pygame.draw.rect(self.screen, COLOR_WHITE, wiring_rect, 3)

    def draw_cell_object(self, x: int, y: int, rect: pygame.Rect):
        pygame.draw.rect(self.screen, COLOR_WHITE, rect)
        obj = self.state.static_objects[y, x]

        if obj != 0:
            # Special handling for barrier: check initially_active state
            bar_idx = self.state.barrier_at(x, y)
            if bar_idx is not None:
                active = self.state.barriers[bar_idx].initially_active
                self._draw_barrier(rect, initially_active=active)
            else:
                self._dispatch_static_draw(obj, rect)

        # Conveyor overlays
        if (y, x) in self.state.item_conveyors:
            self._draw_item_conveyor(rect, self.state.item_conveyors[(y, x)])
        if (y, x) in self.state.player_conveyors:
            self._draw_player_conveyor(rect, self.state.player_conveyors[(y, x)])

        # Barrier index label
        bar_idx = self.state.barrier_at(x, y)
        if bar_idx is not None:
            lbl = self.small_font.render(f"[{bar_idx}]", True, COLOR_WHITE)
            self.screen.blit(lbl, (rect.x + 2, rect.y + 2))

        # Button index label + wiring indicator
        btn_idx = self.state.button_at(x, y)
        if btn_idx is not None:
            btn = self.state.buttons[btn_idx]
            targets_str = ",".join(str(i) for i in btn.target_barrier_idxs) or "?"
            lbl = self.small_font.render(f"→[{targets_str}]", True, COLOR_WHITE)
            self.screen.blit(lbl, (rect.x + 2, rect.y + 2))

        # Pressure plate index label + wiring indicator
        pp_idx = self.state.pressure_plate_at(x, y)
        if pp_idx is not None:
            pp = self.state.pressure_plates[pp_idx]
            targets_str = ",".join(str(i) for i in pp.target_barrier_idxs) or "?"
            lbl = self.small_font.render(f"→[{targets_str}]", True, COLOR_BLACK)
            self.screen.blit(lbl, (rect.x + 2, rect.y + 2))

        # Agent overlay
        if (x, y) in self.state.agent_positions:
            agent_idx = self.state.agent_positions.index((x, y))
            self._draw_agent(rect, agent_idx)

        pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY, rect, 1)

    # ---- Cell sprites ----

    def _draw_wall(self, rect: pygame.Rect):
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        for i in range(rect.x, rect.x + rect.width, 8):
            pygame.draw.line(self.screen, COLOR_LIGHT_GRAY, (i, rect.y), (i, rect.y + rect.height), 1)
        for j in range(rect.y, rect.y + rect.height, 8):
            pygame.draw.line(self.screen, COLOR_LIGHT_GRAY, (rect.x, j), (rect.x + rect.width, j), 1)

    def _draw_pot(self, rect: pygame.Rect):
        pot_rect = pygame.Rect(rect.x + rect.width * 0.1, rect.y + rect.height * 0.33, rect.width * 0.8, rect.height * 0.57)
        pygame.draw.rect(self.screen, COLOR_GRAY, pot_rect)
        lid_rect = pygame.Rect(rect.x + rect.width * 0.1, rect.y + rect.height * 0.21, rect.width * 0.8, rect.height * 0.15)
        pygame.draw.rect(self.screen, COLOR_DARK_GRAY, lid_rect)
        handle_rect = pygame.Rect(rect.x + rect.width * 0.4, rect.y + rect.height * 0.16, rect.width * 0.2, rect.height * 0.08)
        pygame.draw.rect(self.screen, COLOR_DARK_GRAY, handle_rect)

    def _draw_plate_pile(self, rect: pygame.Rect):
        for x, y in [(rect.centerx - rect.width * 0.15, rect.centery - rect.height * 0.15),
                     (rect.centerx + rect.width * 0.15, rect.centery + rect.height * 0.05),
                     (rect.centerx - rect.width * 0.05, rect.centery + rect.height * 0.2)]:
            pygame.draw.circle(self.screen, COLOR_WHITE, (int(x), int(y)), int(rect.width * 0.2))
            pygame.draw.circle(self.screen, COLOR_GRAY, (int(x), int(y)), int(rect.width * 0.2), 1)

    def _draw_goal(self, rect: pygame.Rect):
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        inner = pygame.Rect(rect.x + rect.width * 0.1, rect.y + rect.height * 0.1, rect.width * 0.8, rect.height * 0.8)
        pygame.draw.rect(self.screen, COLOR_GREEN, inner)

    def _draw_recipe_indicator(self, rect: pygame.Rect):
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        inner = pygame.Rect(rect.x + rect.width * 0.1, rect.y + rect.height * 0.1, rect.width * 0.8, rect.height * 0.8)
        pygame.draw.rect(self.screen, COLOR_BROWN, inner)

    def _draw_ingredient_pile(self, rect: pygame.Rect, ingredient_idx: int):
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        color = INGREDIENT_COLORS[ingredient_idx % len(INGREDIENT_COLORS)]
        radius = int(rect.width * 0.15)
        for px, py in [(rect.centerx, rect.y + rect.height * 0.15),
                       (rect.x + rect.width * 0.3, rect.y + rect.height * 0.4),
                       (rect.x + rect.width * 0.8, rect.y + rect.height * 0.35),
                       (rect.x + rect.width * 0.4, rect.y + rect.height * 0.8),
                       (rect.x + rect.width * 0.75, rect.y + rect.height * 0.75)]:
            pygame.draw.circle(self.screen, color, (int(px), int(py)), radius)

    def _draw_item_conveyor(self, rect: pygame.Rect, direction: int = 0):
        pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY, rect)
        for i in range(4):
            y_off = int(rect.y + rect.height * (0.15 + i * 0.2))
            pygame.draw.line(self.screen, COLOR_GRAY, (rect.x + int(rect.width * 0.05), y_off),
                             (rect.x + int(rect.width * 0.95), y_off), 2)
        self._draw_arrow(rect, direction, COLOR_ORANGE)

    def _draw_player_conveyor(self, rect: pygame.Rect, direction: int = 0):
        pygame.draw.rect(self.screen, (173, 216, 230), rect)
        for i in range(4):
            y_off = int(rect.y + rect.height * (0.15 + i * 0.2))
            pygame.draw.line(self.screen, COLOR_BLUE, (rect.x + int(rect.width * 0.05), y_off),
                             (rect.x + int(rect.width * 0.95), y_off), 2)
        self._draw_arrow(rect, direction, COLOR_CYAN)

    def _draw_arrow(self, rect: pygame.Rect, direction: int, color):
        cx, cy = rect.centerx, rect.centery
        s = int(rect.width * 0.2)
        if direction == 2:   pts = [(cx+s,cy),(cx-int(s*.6),cy-int(s*.6)),(cx-int(s*.6),cy+int(s*.6))]
        elif direction == 3: pts = [(cx-s,cy),(cx+int(s*.6),cy-int(s*.6)),(cx+int(s*.6),cy+int(s*.6))]
        elif direction == 0: pts = [(cx,cy-s),(cx-int(s*.6),cy+int(s*.6)),(cx+int(s*.6),cy+int(s*.6))]
        else:                pts = [(cx,cy+s),(cx-int(s*.6),cy-int(s*.6)),(cx+int(s*.6),cy-int(s*.6))]
        pygame.draw.polygon(self.screen, color, pts)

    def _draw_button(self, rect: pygame.Rect):
        """Draw a button tile (interactive switch)."""
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        # Outer ring
        pygame.draw.ellipse(self.screen, COLOR_TEAL, rect.inflate(-6, -6))
        # Inner circle
        inner = rect.inflate(-rect.width // 2, -rect.height // 2)
        inner.center = rect.center
        pygame.draw.ellipse(self.screen, COLOR_WHITE, inner)
        # "BTN" label
        lbl = self.small_font.render("BTN", True, COLOR_DARK_GRAY)
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _draw_barrier(self, rect: pygame.Rect, initially_active: bool = True):
        """Draw a barrier tile (toggleable wall).
        Active barriers look like maroon walls; inactive ones are translucent outlines.
        """
        if initially_active:
            pygame.draw.rect(self.screen, COLOR_MAROON, rect)
            # Crosshatch pattern to distinguish from normal walls
            for i in range(rect.x, rect.x + rect.width, 10):
                pygame.draw.line(self.screen, (220, 60, 60), (i, rect.y), (i + rect.height, rect.y + rect.height), 1)
            lbl = self.small_font.render("BAR", True, COLOR_WHITE)
        else:
            pygame.draw.rect(self.screen, COLOR_WHITE, rect)
            pygame.draw.rect(self.screen, COLOR_MAROON, rect, 3)
            # Dashed interior
            for i in range(0, rect.width, 8):
                pygame.draw.line(self.screen, COLOR_MAROON,
                                 (rect.x + i, rect.y + 4), (rect.x + i, rect.y + rect.height - 4), 1)
            lbl = self.small_font.render("(bar)", True, COLOR_MAROON)
        self.screen.blit(lbl, (rect.x + 2, rect.y + rect.height - 16))

    def _draw_pressure_plate(self, rect: pygame.Rect):
        """Draw a pressure plate tile (floor-level trigger)."""
        # Floor background
        pygame.draw.rect(self.screen, (220, 240, 200), rect)
        # Plate border
        inner = rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, COLOR_LIME, inner, 3)
        # "PP" label
        lbl = self.small_font.render("PP", True, COLOR_DARK_GRAY)
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _draw_agent(self, rect: pygame.Rect, agent_idx: int):
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        cx, cy = rect.centerx, rect.centery
        s = int(rect.width * 0.25)
        pts = [(cx+s,cy),(cx-int(s*.6),cy-int(s*.7)),(cx-int(s*.6),cy+int(s*.7))]
        pygame.draw.polygon(self.screen, color, pts)
        pygame.draw.polygon(self.screen, COLOR_WHITE, pts, 2)
        num = self.small_font.render(str(agent_idx), True, COLOR_BLACK)
        self.screen.blit(num, num.get_rect(center=(cx + int(s*0.3), cy)))

    def _draw_generic(self, rect: pygame.Rect, obj: int):
        pygame.draw.rect(self.screen, COLOR_GRAY, rect.inflate(-4, -4))
        text = self.small_font.render(f"#{obj}", True, COLOR_BLACK)
        self.screen.blit(text, text.get_rect(center=rect.center))

    # ---- Info / Wiring panel ----

    def draw_info_panel(self):
        available = self.window_width - TOOLBAR_WIDTH
        panel_x = TOOLBAR_WIDTH + available // 2
        panel_width = max(INFO_PANEL_WIDTH, self.window_width - panel_x)
        pygame.draw.rect(self.screen, COLOR_DARK_GRAY, (panel_x, TOP_MENU_HEIGHT, panel_width, self.window_height))

        y = TOP_MENU_HEIGHT + 10

        # If a button/pressure-plate is selected, show wiring panel
        if self.wiring_cell is not None:
            wx, wy = self.wiring_cell
            btn_idx = self.state.button_at(wx, wy)
            pp_idx = self.state.pressure_plate_at(wx, wy)
            bar_idx = self.state.barrier_at(wx, wy)

            if btn_idx is not None or pp_idx is not None:
                y = self._draw_wiring_panel(panel_x, panel_width, y, btn_idx, pp_idx)
                return
            if bar_idx is not None:
                y = self._draw_barrier_panel(panel_x, panel_width, y, bar_idx)
                return

        # Default info panel
        title = self.title_font.render("Info", True, COLOR_WHITE)
        self.screen.blit(title, (panel_x + 10, y))
        y += 40

        def draw_text(lines, color=COLOR_WHITE):
            nonlocal y
            for line in lines:
                surf = self.small_font.render(line[:40], True, color)
                self.screen.blit(surf, (panel_x + 10, y))
                y += 20

        draw_text([
            f"Size: {self.state.width}x{self.state.height}",
            f"Agents: {len(self.state.agent_positions)}",
            f"Barriers: {len(self.state.barriers)}",
            f"Buttons: {len(self.state.buttons)}",
            f"Pressure Plates: {len(self.state.pressure_plates)}",
            f"Name: {self.state.layout_name}",
            "",
        ])

        tool = TOOLS[self.state.selected_tool]
        draw_text(["Selected Tool:", f"{tool.symbol} - {tool.name}", tool.description, ""])

        draw_text(["Shortcuts:", "Q=Button  K=Barrier  L=Pressure Plate",
                   "Ctrl+Z=Undo  Ctrl+Y=Redo  Esc=Close wiring", ""])

        # Barrier list
        if self.state.barriers:
            draw_text(["Barriers:"])
            for i, b in enumerate(self.state.barriers):
                state_str = "active" if b.initially_active else "inactive"
                draw_text([f"  [{i}] ({b.y},{b.x}) {state_str}"])

        if self.validation_messages:
            y += 10
            v_title = self.font.render("Validation:", True, COLOR_YELLOW)
            self.screen.blit(v_title, (panel_x + 10, y))
            y += 25
            for msg in self.validation_messages[:10]:
                mc = COLOR_GREEN if msg.startswith("✓") else COLOR_RED
                surf = self.small_font.render(msg[:40], True, mc)
                self.screen.blit(surf, (panel_x + 10, y))
                y += 20

    def _draw_wiring_panel(self, panel_x, panel_width, y, btn_idx, pp_idx):
        """Draw the wiring configuration panel for a selected button or pressure plate.
        Populates self._wiring_action_rects and self._wiring_barrier_rects so that
        _handle_info_panel_click can use the exact rendered positions for hit-testing.
        """
        self._wiring_action_rects = []
        self._wiring_barrier_rects = []

        wx, wy = self.wiring_cell
        is_button = btn_idx is not None
        obj = self.state.buttons[btn_idx] if is_button else self.state.pressure_plates[pp_idx]
        kind = "Button" if is_button else "Pressure Plate"
        pos_str = f"({wy},{wx})"

        # Header
        title = self.title_font.render(f"Wire {kind}", True, COLOR_TEAL if is_button else COLOR_LIME)
        self.screen.blit(title, (panel_x + 10, y))
        y += 32
        sub = self.small_font.render(f"Position: {pos_str}  [Esc to close]", True, COLOR_LIGHT_GRAY)
        self.screen.blit(sub, (panel_x + 10, y))
        y += 24

        # Separator
        pygame.draw.line(self.screen, COLOR_LIGHT_GRAY, (panel_x + 5, y), (panel_x + panel_width - 5, y), 1)
        y += 10

        # --- Action type selector ---
        act_lbl = self.font.render("Action Type:", True, COLOR_WHITE)
        self.screen.blit(act_lbl, (panel_x + 10, y))
        y += 24

        for action_type, label in BUTTON_ACTION_CHOICES:
            selected = (_button_action_value(obj.action_type) == action_type)
            bg = COLOR_TEAL if selected else COLOR_GRAY
            btn_rect = pygame.Rect(panel_x + 10, y, panel_width - 25, 24)
            self._wiring_action_rects.append((btn_rect, action_type))
            pygame.draw.rect(self.screen, bg, btn_rect)
            pygame.draw.rect(self.screen, COLOR_WHITE, btn_rect, 1)
            tag = " ✓" if selected else ""
            txt = self.small_font.render(f"{action_type}: {label}{tag}", True, COLOR_WHITE)
            self.screen.blit(txt, (panel_x + 14, y + 3))
            y += 28

        y += 14

        # Separator
        pygame.draw.line(self.screen, COLOR_LIGHT_GRAY, (panel_x + 5, y), (panel_x + panel_width - 5, y), 1)
        y += 10

        # --- Barrier linker ---
        bar_lbl = self.font.render("Link to Barriers:", True, COLOR_WHITE)
        self.screen.blit(bar_lbl, (panel_x + 10, y))
        y += 24

        if not self.state.barriers:
            no_bar = self.small_font.render("No barriers placed yet.", True, COLOR_LIGHT_GRAY)
            self.screen.blit(no_bar, (panel_x + 14, y))
            y += 22
        else:
            for i, barrier in enumerate(self.state.barriers):
                linked = i in obj.target_barrier_idxs
                # Checkbox — store full row rect for easier clicking
                row_rect = pygame.Rect(panel_x + 10, y, panel_width - 25, 22)
                cb_rect  = pygame.Rect(panel_x + 10, y + 2, 18, 18)
                self._wiring_barrier_rects.append(row_rect)     # ← store for click handler
                pygame.draw.rect(self.screen, COLOR_WHITE, cb_rect)
                pygame.draw.rect(self.screen, COLOR_BLACK, cb_rect, 1)
                if linked:
                    pygame.draw.line(self.screen, COLOR_TEAL, cb_rect.topleft, cb_rect.bottomright, 3)
                    pygame.draw.line(self.screen, COLOR_TEAL, cb_rect.topright, cb_rect.bottomleft, 3)
                state_str = "●" if barrier.initially_active else "○"
                txt = self.small_font.render(f"[{i}] {state_str} ({barrier.y},{barrier.x})", True, COLOR_WHITE)
                self.screen.blit(txt, (panel_x + 34, y + 3))
                y += 26

            y += 8

        return y

    def _draw_barrier_panel(self, panel_x, panel_width, y, bar_idx):
        """Draw the configuration panel for a selected barrier tile.
        Populates self._barrier_toggle_rect for _handle_info_panel_click.
        """
        self._barrier_toggle_rect = None

        barrier = self.state.barriers[bar_idx]
        title = self.title_font.render(f"Barrier [{bar_idx}]", True, COLOR_MAROON)
        self.screen.blit(title, (panel_x + 10, y))
        y += 32
        sub = self.small_font.render(f"Position: ({barrier.y},{barrier.x})  [Esc to close]", True, COLOR_LIGHT_GRAY)
        self.screen.blit(sub, (panel_x + 10, y))
        y += 24

        pygame.draw.line(self.screen, COLOR_LIGHT_GRAY, (panel_x + 5, y), (panel_x + panel_width - 5, y), 1)
        y += 10

        # Toggle initially-active
        toggle_rect = pygame.Rect(panel_x + 10, y, panel_width - 25, 28)
        self._barrier_toggle_rect = toggle_rect                 # ← store for click handler
        state_str = "ACTIVE (solid wall)" if barrier.initially_active else "INACTIVE (open)"
        bg = COLOR_MAROON if barrier.initially_active else COLOR_GRAY
        pygame.draw.rect(self.screen, bg, toggle_rect)
        pygame.draw.rect(self.screen, COLOR_WHITE, toggle_rect, 1)
        txt = self.small_font.render(f"Initially: {state_str}  [click]", True, COLOR_WHITE)
        self.screen.blit(txt, (panel_x + 14, y + 5))
        y += 38

        # Which buttons / plates link to this barrier
        pygame.draw.line(self.screen, COLOR_LIGHT_GRAY, (panel_x + 5, y), (panel_x + panel_width - 5, y), 1)
        y += 10
        linked_lbl = self.small_font.render("Linked by:", True, COLOR_LIGHT_GRAY)
        self.screen.blit(linked_lbl, (panel_x + 10, y))
        y += 20
        found = False
        for i, btn in enumerate(self.state.buttons):
            if bar_idx in btn.target_barrier_idxs:
                t = self.small_font.render(f"  Button {i} at ({btn.y},{btn.x})", True, COLOR_TEAL)
                self.screen.blit(t, (panel_x + 10, y)); y += 18; found = True
        for i, pp in enumerate(self.state.pressure_plates):
            if bar_idx in pp.target_barrier_idxs:
                t = self.small_font.render(f"  Plate {i} at ({pp.y},{pp.x})", True, COLOR_LIME)
                self.screen.blit(t, (panel_x + 10, y)); y += 18; found = True
        if not found:
            t = self.small_font.render("  (nothing)", True, COLOR_LIGHT_GRAY)
            self.screen.blit(t, (panel_x + 10, y)); y += 18

        return y

    # ---- Dialogs ----

    def draw_export_dialog(self):
        self._draw_overlay()
        dw, dh = 600, 300
        dx = (self.window_width - dw) // 2
        dy = (self.window_height - dh) // 2
        pygame.draw.rect(self.screen, COLOR_WHITE, (dx, dy, dw, dh))
        pygame.draw.rect(self.screen, COLOR_BLUE, (dx, dy, dw, dh), 3)
        self.screen.blit(self.title_font.render("Layout Exported!", True, COLOR_BLACK), (dx+20, dy+20))
        self.screen.blit(self.font.render("Saved to:", True, COLOR_BLACK), (dx+20, dy+80))
        self.screen.blit(self.small_font.render(self.export_text[:70], True, COLOR_BLUE), (dx+20, dy+110))
        close = pygame.Rect(dx + dw - 100, dy + dh - 50, 80, 35)
        pygame.draw.rect(self.screen, COLOR_BLUE, close)
        self.screen.blit(self.font.render("Close", True, COLOR_WHITE), self.font.render("Close", True, COLOR_WHITE).get_rect(center=close.center))
        if pygame.mouse.get_pressed()[0] and close.collidepoint(pygame.mouse.get_pos()):
            self.show_export_dialog = False

    def draw_load_dialog(self):
        if not DEPENDENCIES_AVAILABLE:
            return
        self._draw_overlay()
        dw, dh = 500, 500
        dx = (self.window_width - dw) // 2
        dy = (self.window_height - dh) // 2
        pygame.draw.rect(self.screen, COLOR_WHITE, (dx, dy, dw, dh))
        pygame.draw.rect(self.screen, COLOR_BLUE, (dx, dy, dw, dh), 3)
        self.screen.blit(self.title_font.render("Load Layout", True, COLOR_BLACK), (dx+20, dy+20))
        y = dy + 70
        layout_names = list(overcooked_v3_layouts.keys())
        mx, my = pygame.mouse.get_pos()
        for name in layout_names[:15]:
            r = pygame.Rect(dx+20, y, dw-40, 25)
            pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY if r.collidepoint(mx, my) else COLOR_WHITE, r)
            pygame.draw.rect(self.screen, COLOR_GRAY, r, 1)
            self.screen.blit(self.small_font.render(name, True, COLOR_BLACK), (dx+30, y+5))
            if r.collidepoint(mx, my) and pygame.mouse.get_pressed()[0]:
                self.load_layout(name); self.show_load_dialog = False; return
            y += 28
        close = pygame.Rect(dx + dw - 100, dy + dh - 50, 80, 35)
        pygame.draw.rect(self.screen, COLOR_RED, close)
        self.screen.blit(self.font.render("Cancel", True, COLOR_WHITE), self.font.render("Cancel", True, COLOR_WHITE).get_rect(center=close.center))
        if pygame.mouse.get_pressed()[0] and close.collidepoint(mx, my):
            self.show_load_dialog = False

    def draw_resize_dialog(self):
        self._draw_overlay()
        dw, dh = 400, 300
        dx = (self.window_width - dw) // 2
        dy = (self.window_height - dh) // 2
        pygame.draw.rect(self.screen, COLOR_WHITE, (dx, dy, dw, dh))
        pygame.draw.rect(self.screen, COLOR_BLUE, (dx, dy, dw, dh), 3)
        self.screen.blit(self.title_font.render("Resize Grid", True, COLOR_BLACK), (dx+20, dy+20))
        mx, my = pygame.mouse.get_pos()

        y = dy + 80
        for label, val, mode in [("Width:", self.resize_width_input, "width"), ("Height:", self.resize_height_input, "height")]:
            self.screen.blit(self.font.render(label, True, COLOR_BLACK), (dx+30, y))
            inp_rect = pygame.Rect(dx+150, y, 150, 35)
            border_col = COLOR_BLUE if self.resize_input_mode == mode else COLOR_BLACK
            pygame.draw.rect(self.screen, COLOR_WHITE, inp_rect)
            pygame.draw.rect(self.screen, border_col, inp_rect, 2)
            self.screen.blit(self.font.render(val, True, COLOR_BLACK), (inp_rect.x+10, inp_rect.y+5))
            if pygame.mouse.get_pressed()[0] and inp_rect.collidepoint(mx, my):
                self.resize_input_mode = mode
            y += 60

        if self.resize_error:
            self.screen.blit(self.small_font.render(self.resize_error, True, COLOR_RED), (dx+30, dy+200))

        ok = pygame.Rect(dx+80, dy+dh-50, 80, 35)
        cancel = pygame.Rect(dx+240, dy+dh-50, 80, 35)
        pygame.draw.rect(self.screen, COLOR_GREEN, ok)
        pygame.draw.rect(self.screen, COLOR_RED, cancel)
        self.screen.blit(self.font.render("OK", True, COLOR_WHITE), self.font.render("OK", True, COLOR_WHITE).get_rect(center=ok.center))
        self.screen.blit(self.font.render("Cancel", True, COLOR_WHITE), self.font.render("Cancel", True, COLOR_WHITE).get_rect(center=cancel.center))
        pressed = pygame.mouse.get_pressed()[0]
        if pressed:
            if ok.collidepoint(mx, my):
                try:
                    nw = int(self.resize_width_input); nh = int(self.resize_height_input)
                    if not (MIN_GRID_WIDTH <= nw <= MAX_GRID_WIDTH): self.resize_error = f"Width {MIN_GRID_WIDTH}-{MAX_GRID_WIDTH}"; return
                    if not (MIN_GRID_HEIGHT <= nh <= MAX_GRID_HEIGHT): self.resize_error = f"Height {MIN_GRID_HEIGHT}-{MAX_GRID_HEIGHT}"; return
                    self.state.resize(nw, nh); self.show_resize_dialog = False; self._recalc_tile_size()
                except ValueError: self.resize_error = "Enter valid integers"
            elif cancel.collidepoint(mx, my):
                self.show_resize_dialog = False

    def _draw_overlay(self):
        ov = pygame.Surface((self.window_width, self.window_height))
        ov.set_alpha(200)
        ov.fill(COLOR_BLACK)
        self.screen.blit(ov, (0, 0))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _load_jaxmarl_deps()
    _init_tools()  # Build tools after deps are available

    print("=" * 60)
    print("OVERCOOKED V3 LEVEL EDITOR")
    print("=" * 60)
    print()
    if not DEPENDENCIES_AVAILABLE:
        print("WARNING: Running in limited mode without JaxMARL dependencies")
    print("Controls:")
    print("  Left Click: Place  |  Right Click: Erase")
    print("  W=Wall  P=Pot  B=Plate  X=Delivery  A=Agent  E=Erase")
    print("  Q=Button  K=Barrier  L=Pressure Plate")
    print("  0-9=Ingredients  Ctrl+Z=Undo  Ctrl+Y=Redo")
    print("  Ctrl+E=Export  Ctrl+T=Test  Esc=Close wiring panel")
    print()
    print("Wiring workflow:")
    print("  1. Place Barrier tiles (K) – each gets an index [0],[1]...")
    print("  2. Place a Button (Q) or Pressure Plate (L)")
    print("  3. Wiring panel opens on the right – pick action type")
    print("     and tick the barriers you want to link to.")
    print("  4. Click a Barrier tile to see/toggle its initial state")
    print("     and see which buttons/plates are linked to it.")
    print("=" * 60)

    editor = LevelEditor()
    editor.run()


if __name__ == "__main__":
    main()
