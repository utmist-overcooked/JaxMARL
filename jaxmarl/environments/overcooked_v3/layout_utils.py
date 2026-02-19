"""Utilities for working with Overcooked V3 layouts."""

import numpy as np
from typing import List, Tuple, Optional
from jaxmarl.environments.overcooked_v3.common import StaticObject, Direction
from jaxmarl.environments.overcooked_v3.layouts import Layout


def layout_to_string(layout: Layout) -> str:
    """Convert a Layout object back to its string representation.
    
    Args:
        layout: Layout object to convert
        
    Returns:
        String representation suitable for Layout.from_string()
    """
    height, width = layout.static_objects.shape
    
    # Create empty grid with spaces
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Map static objects to symbols
    static_to_symbol = {
        StaticObject.WALL: 'W',
        StaticObject.GOAL: 'X',
        StaticObject.PLATE_PILE: 'B',
        StaticObject.POT: 'P',
        StaticObject.RECIPE_INDICATOR: 'R',
    }
    
    # Item conveyor symbols by direction
    item_conveyor_symbols = {
        Direction.RIGHT: '>',
        Direction.LEFT: '<',
        Direction.UP: '^',
        Direction.DOWN: 'v',
    }
    
    # Player conveyor symbols by direction
    player_conveyor_symbols = {
        Direction.RIGHT: ']',
        Direction.LEFT: '[',
        Direction.UP: '{',
        Direction.DOWN: '}',
    }
    
    # Create lookup dictionaries for conveyor positions
    item_conveyors = {(y, x): direction for y, x, direction in layout.item_conveyor_info}
    player_conveyors = {(y, x): direction for y, x, direction in layout.player_conveyor_info}
    
    # Fill in the grid
    for y in range(height):
        for x in range(width):
            obj = layout.static_objects[y, x]
            
            # Check for item conveyors first
            if (y, x) in item_conveyors:
                direction = item_conveyors[(y, x)]
                grid[y][x] = item_conveyor_symbols[direction]
            # Check for player conveyors
            elif (y, x) in player_conveyors:
                direction = player_conveyors[(y, x)]
                grid[y][x] = player_conveyor_symbols[direction]
            # Check for ingredient piles
            elif StaticObject.is_ingredient_pile(obj):
                ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                grid[y][x] = str(ingredient_idx)
            # Check for other static objects
            elif obj in static_to_symbol:
                grid[y][x] = static_to_symbol[obj]
            # Otherwise leave as space (empty)
    
    # Place agents (agents get placed on top of static objects)
    for agent_x, agent_y in layout.agent_positions:
        grid[agent_y][agent_x] = 'A'
    
    # Convert grid to string
    lines = [''.join(row) for row in grid]
    return '\n' + '\n'.join(lines) + '\n'


def get_layout_info(layout: Layout) -> dict:
    """Get summary information about a layout.
    
    Args:
        layout: Layout object to analyze
        
    Returns:
        Dictionary with layout statistics
    """
    info = {
        'dimensions': (layout.width, layout.height),
        'num_agents': len(layout.agent_positions),
        'num_pots': 0,
        'num_ingredient_piles': {},
        'num_plate_piles': 0,
        'num_goals': 0,
        'num_walls': 0,
        'num_item_conveyors': len(layout.item_conveyor_info),
        'num_player_conveyors': len(layout.player_conveyor_info),
        'has_recipe_indicator': False,
        'possible_recipes': layout.possible_recipes,
    }
    
    # Count objects in static_objects array
    for y in range(layout.height):
        for x in range(layout.width):
            obj = layout.static_objects[y, x]
            
            if obj == StaticObject.POT:
                info['num_pots'] += 1
            elif obj == StaticObject.PLATE_PILE:
                info['num_plate_piles'] += 1
            elif obj == StaticObject.GOAL:
                info['num_goals'] += 1
            elif obj == StaticObject.WALL:
                info['num_walls'] += 1
            elif obj == StaticObject.RECIPE_INDICATOR:
                info['has_recipe_indicator'] = True
            elif StaticObject.is_ingredient_pile(obj):
                ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                if ingredient_idx not in info['num_ingredient_piles']:
                    info['num_ingredient_piles'][ingredient_idx] = 0
                info['num_ingredient_piles'][ingredient_idx] += 1
    
    return info


def validate_layout(layout: Layout) -> Tuple[bool, List[str]]:
    """Validate a layout for common issues.
    
    Args:
        layout: Layout object to validate
        
    Returns:
        Tuple of (is_valid, list of error/warning messages)
    """
    errors = []
    warnings = []
    
    # Get layout info
    info = get_layout_info(layout)
    
    # Check for required elements
    if info['num_agents'] == 0:
        errors.append("Layout must have at least one agent")
    
    if info['num_goals'] == 0:
        errors.append("Layout must have at least one delivery zone (goal)")
    
    if len(info['num_ingredient_piles']) == 0:
        errors.append("Layout must have at least one ingredient pile")
    
    if info['num_plate_piles'] == 0:
        warnings.append("No plate pile found - agents won't be able to serve soup")
    
    if info['num_pots'] == 0:
        warnings.append("No pots found - agents won't be able to cook")
    
    # Check for max limits (from settings.py)
    from jaxmarl.environments.overcooked_v3.settings import MAX_POTS, MAX_ITEM_CONVEYORS, MAX_PLAYER_CONVEYORS
    
    if info['num_pots'] > MAX_POTS:
        errors.append(f"Too many pots ({info['num_pots']} > {MAX_POTS}). Increase MAX_POTS in settings.py")
    
    if info['num_item_conveyors'] > MAX_ITEM_CONVEYORS:
        errors.append(f"Too many item conveyors ({info['num_item_conveyors']} > {MAX_ITEM_CONVEYORS})")
    
    if info['num_player_conveyors'] > MAX_PLAYER_CONVEYORS:
        errors.append(f"Too many player conveyors ({info['num_player_conveyors']} > {MAX_PLAYER_CONVEYORS})")
    
    # Check recipes
    if layout.possible_recipes is None or len(layout.possible_recipes) == 0:
        if not info['has_recipe_indicator']:
            errors.append("Layout has no recipe indicator and no possible_recipes specified")
    else:
        # Validate recipe format
        for i, recipe in enumerate(layout.possible_recipes):
            if not isinstance(recipe, list) or len(recipe) != 3:
                errors.append(f"Recipe {i} must be a list of exactly 3 ingredient indices")
            else:
                # Check if recipe uses ingredients that exist in layout
                for ingredient_idx in recipe:
                    if ingredient_idx not in info['num_ingredient_piles'] and ingredient_idx < layout.num_ingredients:
                        warnings.append(f"Recipe uses ingredient {ingredient_idx} but no pile exists in layout")
    
    # Combine errors and warnings
    all_messages = errors + warnings
    is_valid = len(errors) == 0
    
    return is_valid, all_messages


def annotate_layout_string(layout_string: str) -> str:
    """Add annotations to a layout string explaining the symbols.
    
    Args:
        layout_string: Layout string to annotate
        
    Returns:
        Annotated string with legend
    """
    legend = """
Symbol Legend:
  W = Wall/Counter
  P = Pot
  B = Plate (Bowl) Pile
  X = Delivery Zone (Goal)
  A = Agent Start Position
  R = Recipe Indicator (randomized recipes)
  0-9 = Ingredient Piles (0=onion, 1=tomato, 2=lettuce, etc.)
  
  Item Conveyors (move items):
    > = moves right
    < = moves left
    ^ = moves up
    v = moves down
  
  Player Conveyors (push agents):
    ] = pushes right
    [ = pushes left
    { = pushes up
    } = pushes down
  
  [space] = Empty walkable floor

Layout:
"""
    return legend + layout_string
