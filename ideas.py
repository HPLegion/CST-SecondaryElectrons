import numpy as np
def rotate_about_axis(vector, axis, angle):
    """
    Rotate a vector around a given axis (rooted at origin) and angle (in radian)
    The sense of rotation follows the right hand rule
    The rotation is implemented using the Euler Rodriguez Formula
    See: https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    """
    # Perform a very rough check that vector and axis have the right dimensions
    assert(vector.size == 3)
    assert(axis.size == 3)

    # Compute helper values
    phi = angle / 2
    norm_ax = axis / np.linalg.norm(axis)
    
    # Compute Euler Rodrigues parameters
    a = np.cos(phi)
    b = np.sin(phi) * norm_ax[0]
    c = np.sin(phi) * norm_ax[1]
    d = np.sin(phi) * norm_ax[2]

    # Compose rotation matrix
    rot_mat = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                        [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                        [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
    
    # Perform rotation and return result
    return rot_mat.dot(vector)

def rotate_about_x(vector, angle):
    """
    Rotate a vector around x-axis by given angle (in radian)
    The sense of rotation follows the right hand rule
    """
    return rotate_about_axis(vector, np.array([1, 0, 0]), angle)

def rotate_about_y(vector, angle):
    """
    Rotate a vector around y-axis by given angle (in radian
    The sense of rotation follows the right hand rule
    """
    return rotate_about_axis(vector, np.array([0, 1, 0]), angle)

def rotate_about_z(vector, angle):
    """
    Rotate a vector around z-axis by given angle (in radian
    The sense of rotation follows the right hand rule
    """
    return rotate_about_axis(vector, np.array([0, 0, 1]), angle)
    
def angle_between(vec1, vec2):
    """ 
    Returns the angle between two vectors in radians
    """
    assert(vec1.size == vec2.size)

    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    return np.arccos(v1.dot(v2))

def import_trajectory_file(filename):
    """
    Imports a CST trajectory file, may apply filters to the trajectories immediately while importing,
    returns a list with particle impact trajectories
    (A list of tuples containing the last and 2nd to last recorded position of each particle, this
    should be sufficient to approximate the impact position and angle)
    """
    raise NotImplementedError
    raise RuntimeWarning("May use problem specific import filter")

def intersection_with_model(line, model, atol=1e-6):
    """
    finds the intersection of a line (list of two points) with the provided model object,
    returns a tuple consisting of the collision location and the surface normal vector at that position
    atol defines the maximum distance between line and model to be a valid collision.
    """
    raise NotImplementedError

def generate_secondaries(primary, model):
    """
    Uses the line defined by the end of a primary trajectory and the model to generate secondary electrons
    """
    # Get primary collision information
    raise NotImplementedError

    # Determine number of secondaries to generate (may have to introduce and artifical upper bound)

    # For each generated secondary:

        # Determine kinetic energy (from random distribution)


        # Determine direction (from random distribution)


        # Translate to CST compatible dimensions

        # Append to list of secondaries


def write_secondary_file(filename):
    """
    writes the generated secondary particles to an input file that can be imported into CST
    """
    raise NotImplementedError