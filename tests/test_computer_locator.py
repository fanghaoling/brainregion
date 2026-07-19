"""Locator + ResolvedTarget contract tests."""

from __future__ import annotations

import pytest

from brainregion.computer.contracts import UIElement
from brainregion.computer.locator import (
    ElementDescriptor,
    Locator,
    PanelAnchor,
    ResolvedTarget,
    WithinPanel,
)


_DIGEST = "a" * 64


def test_panel_anchor_requires_one_field():
    with pytest.raises(ValueError):
        PanelAnchor()


def test_element_descriptor_requires_one_field():
    with pytest.raises(ValueError):
        ElementDescriptor()


def test_element_descriptor_matches_role_label_substring_attr_subset():
    element = UIElement(
        element_id="e",
        role="button",
        label="Add Component",
        attributes=(("icon_shape", "plus"),),
    )
    assert ElementDescriptor(
        role="button", label="add comp", attributes=(("icon_shape", "plus"),)
    ).matches(element)
    assert not ElementDescriptor(role="button", label="Save").matches(element)
    assert not ElementDescriptor(role="checkbox", label="add comp").matches(element)
    assert not ElementDescriptor(attributes=(("icon_shape", "minus"),)).matches(element)


def test_within_panel_relation_and_relative_to_are_coupled():
    with pytest.raises(ValueError):
        WithinPanel(relation="below")
    with pytest.raises(ValueError):
        WithinPanel(relative_to=ElementDescriptor(role="x"))
    WithinPanel(relation="below", relative_to=ElementDescriptor(role="x"))


def test_within_panel_band_validated():
    with pytest.raises(ValueError):
        WithinPanel(band="sideways")
    WithinPanel(band="bottom")


def test_locator_from_dict_roundtrip():
    locator = Locator(
        anchor=PanelAnchor(panel_name="Inspector", ordinal="rightmost"),
        within=WithinPanel(band="bottom"),
        descriptor=ElementDescriptor(role="button", label="add component"),
    )
    restored = Locator.from_dict(locator.to_dict())
    assert restored.anchor.panel_name == "inspector"
    assert restored.within.band == "bottom"
    assert restored.descriptor.role == "button"


def test_resolved_target_to_action_intent_binds_same_observation():
    element = UIElement(element_id="ac", role="button", label="Add Component", panel_id="inspector")
    target = ResolvedTarget.from_observation(
        element=element, frame_id="frame-xyz", state_sha256=_DIGEST, available=True, blocker=None
    )
    intent = target.to_action_intent(
        action="click", intent_id="i1", session_id="s", app_id="a"
    )
    assert intent.target_id == "ac"
    assert intent.expected_frame_id == "frame-xyz"
    assert intent.expected_state_sha256 == _DIGEST


def test_resolved_target_available_blocker_invariant():
    with pytest.raises(ValueError):
        ResolvedTarget(
            element_id="e", panel_id=None, frame_id="f", state_sha256=_DIGEST,
            available=True, blocker="below_fold",
        )
    with pytest.raises(ValueError):
        ResolvedTarget(
            element_id="e", panel_id=None, frame_id="f", state_sha256=_DIGEST,
            available=False, blocker=None,
        )
    blocked = ResolvedTarget(
        element_id="e", panel_id=None, frame_id="f", state_sha256=_DIGEST,
        available=False, blocker="below_fold",
    )
    assert blocked.blocker == "below_fold"


def test_resolved_target_invalid_blocker_rejected():
    with pytest.raises(ValueError):
        ResolvedTarget(
            element_id="e", panel_id=None, frame_id="f", state_sha256=_DIGEST,
            available=False, blocker="bogus",
        )
