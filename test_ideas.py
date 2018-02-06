import numpy as np
from numpy.testing import assert_allclose
from ideas import *
import pytest

# Some useful vectors for testing geometry issues
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
ez = np.array([0, 0, 1])

class TestRotationFunctions:
    """
    Contains tests for the rotation functions
    """
    def test_rotate_about_x(self):
        """Tests for rotate_about_x"""
        assert_allclose(rotate_about_x(ex, np.pi/2), ex)
        assert_allclose(rotate_about_x(ey, np.pi/2), ez)
        assert_allclose(rotate_about_x(ez, np.pi/2), -1*ey)
        assert_allclose(rotate_about_x(ey, np.pi/4), (ey+ez)/np.sqrt(2))

    def test_rotate_about_y(self):
        """Tests for rotate_about_y"""
        assert_allclose(rotate_about_y(ex, np.pi/2), -1*ez)
        assert_allclose(rotate_about_y(ey, np.pi/2), ey)
        assert_allclose(rotate_about_y(ez, np.pi/2), ex)
        assert_allclose(rotate_about_y(ez, np.pi/4), (ez+ex)/np.sqrt(2))

    def test_rotate_about_z(self):
        """Tests for rotate_about_z"""
        assert_allclose(rotate_about_z(ex, np.pi/2), ey)
        assert_allclose(rotate_about_z(ey, np.pi/2), -1*ex)
        assert_allclose(rotate_about_z(ez, np.pi/2), ez)
        assert_allclose(rotate_about_z(ex, np.pi/4), (ex+ey)/np.sqrt(2))

    def test_rotate_about_axis(self):
        """Tests for rotate_about_axis"""
        ax = np.array([1, 1, 0])
        assert_allclose(rotate_about_axis(ax, ax, np.pi/2), ax)
        assert_allclose(rotate_about_axis(ex, ax, np.pi), ey, atol=1.e-8)
        assert_allclose(rotate_about_axis(ey, ax, np.pi), ex, atol=1.e-8)
        assert_allclose(rotate_about_axis(ez, ax, np.pi/2), (ex-ey)/np.sqrt(2), atol=1.e-8)

def test_angle_between():
    """Test the angle_between two vectors function"""
    assert_allclose(angle_between(ex, ey), np.pi/2)
    assert_allclose(angle_between(ey, ez), np.pi/2)
    assert_allclose(angle_between(ey+ez, ez), np.pi/4)

def test_create_line():
    """Tests the create_line function"""
    start = np.array([1, 2, 3])
    stop = np.array([4, 5, 6])
    wrong = np.array([1, 2])
    with pytest.raises(AssertionError):
        create_line(start, start)
    with pytest.raises(AssertionError):
        create_line(start, wrong)
    line = create_line(start, stop)
    assert_allclose(np.array(line.firstVertex().Point), start)
    assert_allclose(np.array(line.lastVertex().Point), stop)
