import pytest

from .core import add, system_exit_1


def test_add():
    assert add(1, 2) == 3


def test_system_exit_1():
    with pytest.raises(SystemExit):
        system_exit_1()
