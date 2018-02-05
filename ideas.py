import numpy as np
def rotate_about_axis(vector, axis, angle):
    """
    Rotate a vector around a given axis and angle (in radian)
    The sense of rotation follows the right hand rule
    The rotation is implemented using the Euler Rodriguez Formula
    See: https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    """
    phi = angle / 2
    norm_ax = axis / np.linalg.norm(axis)
    
    a = np.cos(phi)
    b = np.sin(phi) * norm_ax[0]
    c = np.sin(phi) * norm_ax[1]
    d = np.sin(phi) * norm_ax[2]

    rot_mat = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                        [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                        [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
    print(rot_mat)
    print(vector)
    print(rot_mat.dot(vector))
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
    

ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
ez = np.array([0, 0, 1])

print(rotate_about_x(ex, np.pi/2))
print(rotate_about_x(ey, np.pi/2))
print(rotate_about_x(ez, np.pi/2))