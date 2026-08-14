import pytest

from open_vocab_grasping.planning.filtering import approach_distance_options


def test_approach_distance_options_stay_within_eight_to_ten_centimetres() -> None:
    assert approach_distance_options(0.10, 0.08, 0.01) == pytest.approx([0.10, 0.09, 0.08])


def test_approach_distance_options_reject_invalid_range() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        approach_distance_options(0.08, 0.10, 0.01)
