"""
This document contains a collection of functions to help with the creation of secondary electrons
for particle collisions with model surfaces for CST simulation studio
"""
######################################################################
###Import Statements
######################################################################

# Trivial import statements
import sys
import warnings
import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
import pandas as pd
PI = np.pi
#Load FreeCAD
FREECADPATH = "C:/Anaconda3/pkgs/freecad-0.17-py36_11/Library/bin"
sys.path.append(FREECADPATH)
try:
    from FreeCAD import Part
except ImportError:
    print("Could not find the FreeCAD module. Check the FREECADPATH variable!")
    print("Current path is:", FREECADPATH)
    exit()

######################################################################
###Coordinate Transformations and Vector Geometry
######################################################################
def normalise(vec):
    """
    Normalises a vector using the numpy linalg norm function
    """
    return vec/np.linalg.norm(vec)

def rotate_about_axis(vector, axis, angle):
    """
    Rotate a vector around a given axis (rooted at origin) and angle (in radian)
    The sense of rotation follows the right hand rule
    The rotation is implemented using the Euler Rodriguez Formula
    See: https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    vector and axis should be numpy arrays of length 3
    """
    # Perform a check that vector and axis have the right dimensions
    assert vector.shape == (3, ) or vector.shape == (1, 3)
    assert axis.shape == (3, ) or axis.shape == (1, 3)

    # Compute helper values
    phi = angle / 2
    axis = normalise(axis)

    # Compute Euler Rodrigues parameters
    a = np.cos(phi)
    b = np.sin(phi) * axis[0]
    c = np.sin(phi) * axis[1]
    d = np.sin(phi) * axis[2]

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

def angle_between(vec1, vec2, force_acute=False):
    """
    Returns the angle between two vectors in radians
    If force_acute is set to True, the angles are folded into the interval [0,pi/2],
    i.e. alpha>pi/2 ==> pi-alpha
    """
    # Perform a check that vector and axis have the right dimensions
    assert vec1.shape == (3, ) or vec1.shape == (1, 3)
    assert vec2.shape == (3, ) or vec2.shape == (1, 3)

    vec1 = normalise(vec1)
    vec2 = normalise(vec2)
    phi = np.arccos(vec1.dot(vec2))
    if force_acute and phi > PI/2:
        phi = PI - phi
    return phi


######################################################################
###FreeCAD interfacing methods
######################################################################

def load_model(filename):
    """
    Loads a CAD model using the FreeCAD Library. Returns a reference to the FreeCAD.Part object
    """
    return Part.read(filename)

def create_line(start, stop):
    """
    Creates and returns a FreeCAD line from a given set of start and stop coordinates
    Expects start and stop to be numpy arrays
    """
    assert start.shape == (3, ) or start.shape == (1, 3)
    assert stop.shape == (3, ) or stop.shape == (1, 3)
    assert any(np.not_equal(start, stop)), "Start and stop coordinate should not be identical"

    start = tuple(start)
    stop = tuple(stop)
    return Part.makeLine(start, stop)

def intersection_with_model(line, model, atol=1e-6):
    """
    !!!ONLY USE WITH SIMPLE CAD MODELS FOR NOW!!!
    Finds the intersection of a FreeCAD line object with the provided model object
    Returns a tuple consisting of the collision location and the surface normal vector at that
    position
    atol defines the maximum distance between line and model to be a valid collision.
    """
    # Find the intersection of the line and the model by using the shortest distance in between
    dist_info = model.Shells[0].distToShape(line)

    # Check the minimum distance
    distance = dist_info[0]
    if distance > atol:
        raise ValueError("No intersection was found using the given tolerance:", atol)

    # Check the intersection vertex
    inters_vertex = dist_info[1]
    if len(inters_vertex) > 1: #Check if there was more than one intersection
        raise ValueError("More than one possible intersection was found.")
    inters_coord = np.array(inters_vertex[0][0]) # Extract coordinates of intersection

    # Generate the surface normal vector
    inters_geom = dist_info[2][0] # Extract geometry feature at collision point
    if inters_geom[0] == b'Face' or inters_geom[0] == "Face": # Assert the collision is on a face
        face = model.Faces[inters_geom[1]]
        u = inters_geom[2][0]
        v = inters_geom[2][1]
        inters_norm = np.array(face.normalAt(u, v)) # Compute the normal vector
        inters_norm = normalise(inters_norm) #Normalise normal vector
    else:
        raise ValueError("Did not collide on a face geometry, cannot reconstruct normal vec")

    return (inters_coord, inters_norm)

######################################################################
###CST Import Export
######################################################################

def import_trajectory_file(filename):
    """
    Imports a CST trajectory file
    returns a list with particle impact trajectories
    (A list of tuples containing the last and 2nd to last recorded position of each particle, this
    should be sufficient to approximate the impact position and angle)
    """
    #List of all impact data
    impacts = []

    with open(filename, 'r') as inp:
        # Skip 7 rows
        for dummy in range(7):
            inp.readline()

        # Process each line in inp
        last_particle_id = None
        imp_info = dict()
        pos_prior = pos_impact = mom_impact = [None, None, None]
        time_impact = None
        for line in inp:
            # Break on empty line(last line)
            if line == "":
                break

            data = line.split()
            # If reached new particle, update constants
            if data[10] != last_particle_id or line == "":
                #store previous data set
                if last_particle_id is not None:
                    imp_info["mom_impact"] = np.array(mom_impact, dtype=float)
                    imp_info["pos_impact"] = np.array(pos_impact, dtype=float)
                    imp_info["pos_prior"] = np.array(pos_prior, dtype=float)
                    imp_info["time_impact"] = float(time_impact)
                    impacts.append(imp_info)
                    imp_info = dict()

                # extract new constants
                imp_info["mass"] = float(data[6])
                imp_info["charge"] = float(data[7])
                imp_info["macro_charge"] = float(data[8])
                imp_info["particle_id"] = int(data[10])
                imp_info["source_id"] = int(data[11])

                # update last_particle_id
                last_particle_id = data[10]

            # Cache trajectory for this step and the prior step
            pos_prior = pos_impact
            pos_impact = [data[0], data[1], data[2]]
            mom_impact = [data[3], data[4], data[5]]
            time_impact = data[9]
    # Store last particle
    imp_info["mom_impact"] = np.array(mom_impact, dtype=float)
    imp_info["pos_impact"] = np.array(pos_impact, dtype=float)
    imp_info["pos_prior"] = np.array(pos_prior, dtype=float)
    imp_info["time_impact"] = float(time_impact)
    impacts.append(imp_info)
    # Transform pos to mm
    for imp in impacts:
        imp["pos_impact"] = 1000 * imp["pos_impact"]
        imp["pos_prior"] = 1000 * imp["pos_prior"]
    # Return list
    return impacts

def write_secondary_file(filename, particles):
    """
    writes the generated secondary particles to an input file that can be imported into CST
    """
    colnames = ["x", "y", "z", "px", "py", "pz", "mass", "charge", "current"]
    output = pd.DataFrame(particles, columns=colnames)
    output.x = output.x / 1000 #Adjust spatial dimensions from mm to m
    output.y = output.y / 1000
    output.z = output.z / 1000
    output.to_csv(filename, sep=" ", index=False, header=False)

######################################################################
###Secondary Generation
######################################################################

def out_direction(normal, theta, phi):
    """
    A helper function that computes the output direction given a surface normal vector,
    an emission angle theta, and a polar angle phi.
    The polar coordinate has no guaranteed zero-location in this function! It assumes that the
    polar orientation will be uniformely random, and that hence the zero-loc does not matter!
    """
    EX = np.array([1, 0, 0])
    EY = np.array([0, 1, 0])
    normal = normalise(normal)

    # Generate an axis that lies in the plane orthogonal to the surface normal
    # Form cross product with x axis
    axis = np.cross(normal, EX)
    # For numerical stabilty take cross y axis if norm(axis)<0.1, i.e. normal almost parallel to x
    if np.linalg.norm(axis) < 0.1: axis = np.cross(normal, EY)
    axis = normalise(axis)

    #Rotate the axis around the normal by the random angle phi
    rotated_axis = rotate_about_axis(axis, normal, phi)
    #Rotate the normal vector around the new axis to get emission direction
    direct_out = rotate_about_axis(normal, rotated_axis, theta)
    return direct_out

def generate_particle(start, direction, kin_energy, charge, current, mass, relativistic=False):
    """
    Generates a dictionary with particle properties as required by CST, input needs SI Units
    Direction can be a unitless vector of any length, it will be normalised
    relativistic should not be used for low energies as floating point errors may be induced
    """
    C_0 = scipy.constants.speed_of_light
    particle = dict()
    particle["mass"] = mass
    particle["charge"] = charge
    particle["current"] = current
    particle["x"] = start[0]
    particle["y"] = start[1]
    particle["z"] = start[2]
    #Compute momentum
    rest_energy = mass * C_0**2
    if kin_energy > 0.1 * rest_energy and not relativistic:
        warnings.warn("Using non relativistic calculation for particles with E_kin > 10% E_mass",
                      UserWarning)
    if relativistic:
        tot_energy = kin_energy + rest_energy
        abs_momentum = np.sqrt(tot_energy**2 - rest_energy**2)/C_0
    else:
        abs_momentum = np.sqrt(2*mass*kin_energy)
    normed_momentum = abs_momentum / mass / C_0
    direction = normalise(direction)
    particle["px"] = normed_momentum * direction[0]
    particle["py"] = normed_momentum * direction[1]
    particle["pz"] = normed_momentum * direction[2]
    return particle

def generate_electron(start, direction, kin_energy, current=None, relativistic=False):
    """
    Generates a dictionary with electron properties as required by CST, input needs SI Units
    Direction can be a unitless vector of any length, it will be normalised
    relativistic should not be used for low energies as floating point errors may be induced
    """
    q_e = -1*scipy.constants.elementary_charge
    m_e = scipy.constants.electron_mass
    if current is None:
        current = q_e
    return generate_particle(start, direction, kin_energy, q_e, current, m_e, relativistic)

def generate_secondaries(primary, model, base_yield=5, temperature=10):
    """
    Uses the line defined by the end of a primary trajectory and the model to generate
    secondary electrons
    primary: impacting particle
    model: the freecad part
    BASEYIELD: mean number of generated electrons for normal incidence
    temperature: in eV Electron temperature, i.e. mode of gamma distribution
    """
    secondaries = [] # Collection of the secondary electrons

    # Get primary collision information
    impact_line = create_line(primary["pos_prior"], primary["pos_impact"])
    impact_coord, impact_norm = intersection_with_model(impact_line, model)

    # Determine number of secondaries to generate
    # (may have to introduce an artifical upper bound)
    # wrap theta_in to angles smaller than pi/2 (if vectors were showing in opposite directions)
    theta_in = angle_between(primary["mom_impact"], impact_norm, force_acute=True)
    #Artifically clip impact angle to clip electron yield
    if theta_in > 80/180*PI: theta_in = 80/180*PI
    #Scale base yield depending on impact angle
    ang_yield = base_yield/np.cos(theta_in)
    # Number of electrons from poissondist with mean=ang_yield
    num_el = np.random.poisson(ang_yield)

    # For each generated secondary:
    for _ in range(num_el):
        # Determine kinetic energy (from random distribution)
        e_kin = np.random.gamma(2, temperature) # Gamma Dist with shape 2 and Mode at temperature
        e_kin *= scipy.constants.elementary_charge

        # Determine direction (from random distribution)
        phi_out = np.random.uniform(0, 2*PI)
        theta_out = np.arcsin(np.random.uniform() - 1) # cos-dist
        direct_out = out_direction(impact_norm, theta_out, phi_out)

        # Translate to CST compatible dimensions
        electron = generate_electron(impact_coord, direct_out, e_kin)

        # Append to list of secondaries
        secondaries.append(electron)

    return secondaries

######################################################################
###Eval Routines
######################################################################

def filter_particles(particles):
    """
    Function to include usefull filter statements for the primary particle list to remove this
    logic from other routines
    Filter Criteria may change arbitrarily depending on the scenario under investigation
    Takes a list of particles and returns a list of particles passing the filters
    """
    particles_filtered = []
    for par in particles:
        if (par['pos_impact'][2] < 10 and
                -15 < par['pos_impact'][0] < 15 and
                -15 < par['pos_impact'][1] < 15):
            particles_filtered.append(par)
    return particles_filtered

def main_routine(modelfile, trackfile, outfile):
    """The main routine managing the import, secondary creation and export"""
    model = load_model(modelfile)
    particles = import_trajectory_file(trackfile)
    num_inp_par = len(particles)
    particles = filter_particles(particles)

    secondaries = []
    failed = []
    total = len(particles)
    k = 1
    for par in particles:
        try:
            secondaries += generate_secondaries(par, model, base_yield=9)
        except ValueError as error:
            failed.append(par)
            print("Particle ID:", par["particle_id"], error.args)
        print("Progress", k, "of", total, "primaries processed")
        k += 1

    print(num_inp_par, "primary particles imported")
    print(len(particles), "primary particles after filtering")
    print(len(failed), "filtered primaries did not collide")
    print(len(secondaries), "secondaries generated")

    write_secondary_file(outfile, secondaries)

def visual_benchmark(modelfile, trackfile):
    """
    Function for implementing quick visual checks of the secondary distribtuion
    subject to usecase changes
    """
    model = load_model(modelfile)
    particles = import_trajectory_file(trackfile)
    num_inp_par = len(particles)
    particles = filter_particles(particles)

    secondaries = []
    failed = []
    total = len(particles)
    k = 1
    for par in particles:
        try:
            secondaries += generate_secondaries(par, model, base_yield=9)
        except ValueError as error:
            failed.append(par)
            print("Particle ID:", par["particle_id"], error.args)
        print("Progress", k, "of", total, "primaries processed")
        k += 1

   
    print(num_inp_par, "primary particles imported")
    print(len(particles), "primary particles after filtering")
    print(len(failed), "filtered primaries did not collide")
    print(len(secondaries), "secondaries generated")

    angles = []
    for s in secondaries:
        direc = np.array([s["px"], s["py"], s["pz"]])
        angles.append(angle_between(direc, np.array([0, 0, 1])))
    plt.hist(angles, bins=50, density=True)
    plt.show()

# MODELFILE = "./sample/simple_cylinder.stp"
# TRACKFILE = "./sample/cyl.txt"
# OUTFILE = "./sample/sec_cyl.pid"

# MODELFILE = "M:/HANNES CST FILES/GriddedLensTest/mesh_big_3mm.stp"
# TRACKFILE = "M:/HANNES CST FILES/GriddedLensTest/prim_3mm.txt"
# OUTFILE = "M:/HANNES CST FILES/GriddedLensTest/sec_3mm.pid"

# main_routine(MODELFILE, TRACKFILE, OUTFILE)
# #visual_benchmark(MODELFILE, TRACKFILE)
