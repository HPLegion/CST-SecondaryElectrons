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

def test_load_model():
    """Loads a test model"""
    
    mdl = load_model("./test_resources/simple_cylinder.stp")
    assert mdl.Volume == pytest.approx(2*1*np.pi) # Check cylinder volume as simple test

class TestIntersectionWithModel():
    """
    Tests the intersection function
    Based on the model of a cylinder with radius on parallel to x axis
    reaching from x=-1 to x=1
    """
    def __init__(self):
        self.mdl = load_model("./test_resources/simple_cylinder.stp")

    def test_no_intersection(self):
        with pytest.raises(ValueError) as excinfo:
            line = create_line(np.array([2, 2, 2]), np.array([3, 3, 3]))
            intersection_with_model(line, self.mdl)
        assert "No intersection" in str(excinfo.value)

    def test_two_intersections(self):
        with pytest.raises(ValueError) as excinfo:
            line = create_line(np.array([0.1, 0.1, -2]), np.array([0.1, 0.1, 2]))
            intersection_with_model(line, self.mdl)
        assert "More than one" in str(excinfo.value)

    def test_not_on_face(self):
        with pytest.raises(ValueError) as excinfo:
            line = create_line(np.array([1, 0, 2]), np.array([1, 0., 1]))
            intersection_with_model(line, self.mdl)
        assert "on a face" in str(excinfo.value)

    def test_collision_coordinate(self):
        # Test the intersection coordinate for the simple cylinder geometry
        ys = np.arange(-0.92, 0.92, 0.1)
        for y in ys:
            line = create_line(np.array([0, y, 2]), np.array([0, y, 0]))
            coord, norm = intersection_with_model(line, self.mdl)
            assert coord[2] == pytest.approx(np.sqrt(1-y**2))
     
    def test_collision_normal(self):
        # Test the intersection normal for the simple cylinder geometry
        ys = np.arange(-0.92, 0.92, 0.1)
        for y in ys:
            line = create_line(np.array([0, y, -2]), np.array([0, y, 0]))
            coord, norm = intersection_with_model(line, self.mdl)
            assert norm[2] == pytest.approx(np.sin(y))
            assert norm[3] == pytest.approx(np.cos(y))

def test_import_trajectory_file():
    """Tests the import function"""
    imp = import_trajectory_file("./test_resources/test_traj.txt")
    assert len(imp) == 4
    for x in imp:
        assert len(x) == 9
    assert imp[0]["particle_id"] == 0
    assert imp[3]["time_impact"] == pytest.approx(7.7611890e+02)
    assert_allclose(imp[2]["pos_impact"], np.array([-9.5142633e-02, 3.9006030e-04, 5.2500004e-01]))
    assert_allclose(imp[2]["pos_prior"], np.array([-9.4913870e-02, 3.9009575e-04, 5.2436417e-01]))
    assert_allclose(imp[1]["mom_impact"], np.array([-7.8177312e-04, 9.0218059e-08, 2.1720929e-03]))
