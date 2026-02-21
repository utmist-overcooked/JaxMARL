"""Tests for layout utility functions."""

import pytest
from jaxmarl.environments.overcooked_v3.layouts import Layout, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.layout_utils import (
    layout_to_string,
    get_layout_info,
    validate_layout,
    annotate_layout_string,
)


class TestLayoutToString:
    """Test layout to string conversion."""

    def test_round_trip_cramped_room(self):
        """Test that cramped_room can be converted to string and back."""
        # Get original layout
        original = overcooked_v3_layouts["cramped_room"]

        # Convert to string
        layout_str = layout_to_string(original)

        # Convert back to Layout
        reconstructed = Layout.from_string(
            layout_str, possible_recipes=original.possible_recipes
        )

        # Check dimensions match
        assert reconstructed.width == original.width
        assert reconstructed.height == original.height

        # Check agent positions match (might be in different order)
        assert len(reconstructed.agent_positions) == len(original.agent_positions)
        assert set(reconstructed.agent_positions) == set(original.agent_positions)

        # Check static objects match
        assert (reconstructed.static_objects == original.static_objects).all()

    def test_round_trip_conveyor_demo(self):
        """Test that conveyor_demo layout can be converted and reconstructed."""
        original = overcooked_v3_layouts["conveyor_demo"]

        layout_str = layout_to_string(original)
        reconstructed = Layout.from_string(
            layout_str, possible_recipes=original.possible_recipes
        )

        # Check conveyor info preserved
        assert len(reconstructed.item_conveyor_info) == len(original.item_conveyor_info)
        assert set(reconstructed.item_conveyor_info) == set(original.item_conveyor_info)

    def test_round_trip_player_conveyor(self):
        """Test that player_conveyor layouts preserve conveyor info."""
        original = overcooked_v3_layouts["player_conveyor_loop"]

        layout_str = layout_to_string(original)
        reconstructed = Layout.from_string(
            layout_str, possible_recipes=original.possible_recipes
        )

        # Check player conveyor info preserved
        assert len(reconstructed.player_conveyor_info) == len(
            original.player_conveyor_info
        )
        assert set(reconstructed.player_conveyor_info) == set(
            original.player_conveyor_info
        )

    def test_round_trip_all_registered_layouts(self):
        """Test round-trip conversion for all registered layouts."""
        for layout_name, original in overcooked_v3_layouts.items():
            layout_str = layout_to_string(original)
            reconstructed = Layout.from_string(
                layout_str, possible_recipes=original.possible_recipes
            )

            # Basic checks
            assert reconstructed.width == original.width, f"Failed for {layout_name}"
            assert reconstructed.height == original.height, f"Failed for {layout_name}"
            assert len(reconstructed.agent_positions) == len(
                original.agent_positions
            ), f"Failed for {layout_name}"
            assert (reconstructed.static_objects == original.static_objects).all(), (
                f"Failed for {layout_name}"
            )


class TestGetLayoutInfo:
    """Test layout information extraction."""

    def test_cramped_room_info(self):
        """Test info extraction for cramped_room."""
        layout = overcooked_v3_layouts["cramped_room"]
        info = get_layout_info(layout)

        assert info["dimensions"] == (layout.width, layout.height)
        assert info["num_agents"] == 2
        assert info["num_pots"] == 1
        assert info["num_goals"] == 1
        assert info["num_plate_piles"] == 1
        assert 0 in info["num_ingredient_piles"]  # Has ingredient 0 (onion)

    def test_conveyor_demo_info(self):
        """Test info extraction for layout with conveyors."""
        layout = overcooked_v3_layouts["conveyor_demo"]
        info = get_layout_info(layout)

        assert info["num_item_conveyors"] > 0

    def test_player_conveyor_info(self):
        """Test info extraction for player conveyor layout."""
        layout = overcooked_v3_layouts["player_conveyor_loop"]
        info = get_layout_info(layout)

        assert info["num_player_conveyors"] > 0


class TestValidateLayout:
    """Test layout validation."""

    def test_valid_layout(self):
        """Test that registered layouts pass validation."""
        layout = overcooked_v3_layouts["cramped_room"]
        is_valid, messages = validate_layout(layout)

        # Should be valid (though may have warnings)
        assert is_valid, f"cramped_room should be valid, got: {messages}"

    def test_missing_agents(self):
        """Test that layout without agents is rejected at construction."""
        layout_str = """
WWPWW
0   0
W   W
WBWXW
"""
        with pytest.raises(
            ValueError, match="At least one agent position must be provided"
        ):
            Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])

    def test_missing_goal(self):
        """Test that layout without delivery zone fails validation."""
        layout_str = """
WWPWW
0A A0
W   W
WBWWW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])
        is_valid, messages = validate_layout(layout)

        assert not is_valid
        assert any(
            "goal" in msg.lower() or "delivery" in msg.lower() for msg in messages
        )

    def test_missing_ingredients(self):
        """Test that layout without ingredients fails validation."""
        layout_str = """
WWPWW
WA AW
W   W
WBWXW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])
        is_valid, messages = validate_layout(layout)

        assert not is_valid
        assert any("ingredient" in msg.lower() for msg in messages)


class TestAnnotateLayout:
    """Test layout annotation."""

    def test_annotation_includes_legend(self):
        """Test that annotation adds a legend."""
        layout_str = """
WWPWW
0A A0
W   W
WBWXW
"""
        annotated = annotate_layout_string(layout_str)

        assert "Symbol Legend" in annotated
        assert "W = Wall" in annotated
        assert "P = Pot" in annotated
        assert layout_str in annotated
