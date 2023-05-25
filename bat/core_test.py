import numpy as np

from .core import bat, egg_crate


def test_egg_crate():
    assert egg_crate(1, 1) == 37.40367091367856


def test_bat():
    _, fitness_min, _, iterations = bat()
    assert fitness_min < 1
    assert iterations > 0
