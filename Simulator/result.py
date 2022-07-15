# The result of our simulation
# Contains the following data:
# - List of time steps
# - List of nodal displacements
class Result:
    def __init__(self, time_steps, nodal_displacements, nodal_velocities, nodal_accelerations, Es):
        self.time_steps = time_steps
        self.nodal_accelerations = nodal_accelerations
        self.nodal_velocities = nodal_velocities
        self.nodal_displacements = nodal_displacements
        self.Es = Es

    def get_time_steps(self):
        return self.time_steps

    def get_nodal_displacements(self):
        return self.nodal_displacements

    def get_nodal_velocities(self):
        return self.nodal_velocities

    def get_nodal_accelerations(self):
        return self.nodal_accelerations