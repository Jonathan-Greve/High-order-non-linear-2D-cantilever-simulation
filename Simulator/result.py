# The result of our simulation
# Contains the following data:
# - List of time steps
# - List of nodal displacements
class Result:
    def __init__(self, time_steps, nodal_displacements, nodal_velocities, nodal_accelerations, Es, Ms):
        self.time_steps = time_steps
        self.nodal_accelerations = nodal_accelerations
        self.nodal_velocities = nodal_velocities
        self.nodal_displacements = nodal_displacements
        self.Es = Es
        self.Ms = Ms