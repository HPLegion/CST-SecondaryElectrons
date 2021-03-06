import numpy as np
from numpy.testing import assert_allclose
from secondary_electrons import *
import pytest

# Some useful vectors for testing geometry issues
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
ez = np.array([0, 0, 1])

def test_normalise():
    """Test the vector normalisation funcion"""
    for k in range(10):
        n = np.random.randint(1,10)
        vec = np.random.random(n)
        vec = normalise(vec)
        assert np.linalg.norm(vec) == pytest.approx(1)

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
    assert_allclose(angle_between(ey+ez, -1*ez, force_acute=True), np.pi/4)
    assert_allclose(angle_between(ey+ez, -1*ez, force_acute=False), 3*np.pi/4)

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
    mdl = load_model("./test_resources/simple_cylinder.stp")

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
            line = create_line(np.array([0, y, 2]), np.array([0, y, 0]))
            coord, norm = intersection_with_model(line, self.mdl)
            print(norm)
            assert norm[1] == pytest.approx(y)
            assert norm[2] == pytest.approx(np.sqrt(1-y**2))

def test_import_trajectory_file():
    """Tests the import function"""
    imp = import_trajectory_file("./test_resources/test_traj.txt")
    assert len(imp) == 4
    for x in imp:
        assert len(x) == 9
    assert imp[0]["particle_id"] == 0
    assert imp[3]["time_impact"] == pytest.approx(7.7611890e+02)
    assert_allclose(imp[2]["pos_impact"], 1000*np.array([-9.5142633e-02, 3.9006030e-04, 5.2500004e-01]))
    assert_allclose(imp[2]["pos_prior"], 1000*np.array([-9.4913870e-02, 3.9009575e-04, 5.2436417e-01]))
    assert_allclose(imp[1]["mom_impact"], np.array([-7.8177312e-04, 9.0218059e-08, 2.1720929e-03]))

def test_generate_particle():
    """Tests the CST-style particle generation function"""
    C_0 = 299792458
    start = np.array([1, 2, 3])
    direction = np.array([3, 4, 12]) #Length 13
    kin_energy = 13**2/2.0
    mass = 1
    charge = -1
    current = -2
    par = generate_particle(start, direction, kin_energy, charge, current, mass)
    assert par["mass"] == mass
    assert par["charge"] == charge
    assert par["current"] == current
    assert par["x"] == start[0]
    assert par["y"] == start[1]
    assert par["z"] == start[2]
    assert C_0*par["px"] == pytest.approx(3)
    assert C_0*par["py"] == pytest.approx(4)
    assert C_0*par["pz"] == pytest.approx(12)
    #Test high energy warning
    with pytest.warns(UserWarning):
        generate_particle(start, direction, C_0**2, charge, current, mass)

def test_generate_electron():
    """Test the electron generation basics"""
    start = np.array([1, 2, 3])
    direction = np.array([3, 4, 12]) #Length 13
    kin_energy = 13**2/2.0
    par = generate_electron(start, direction, kin_energy)
    assert par["charge"] == pytest.approx(1.60217662e-19)
    assert par["current"] == pytest.approx(1.60217662e-19)
    assert par["mass"] == pytest.approx(9.10938356e-31)

def test_write_secondary_file():
    """Test the output routine"""
    C_0 = 299792458
    start = np.array([1, 2, 3])
    direction = np.array([3, 4, 12]) #Length 13
    kin_energy = 13**2/2.0
    mass = 1
    charge = -1
    current = -2
    par = generate_particle(start, direction, kin_energy, charge, current, mass)
    par2 = generate_particle(start, direction, kin_energy, charge, current, 2*mass)
    write_secondary_file("./test_resources/temp.txt", [par, par2])
    with open("./test_resources/temp.txt") as f:
        assert f.readline() == "0.001 0.002 0.003 1.0006922855944563e-08 1.3342563807926083e-08 4.002769142377825e-08 1 -1 -2\n"
        assert f.readline() == "0.001 0.002 0.003 7.075963010249053e-09 9.434617346998737e-09 2.8303852040996212e-08 2 -1 -2\n"