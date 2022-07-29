# The result of our simulation
# Contains the following data:
# - List of time steps
# - List of nodal displacements
class Result:
    def __init__(self, time_steps, nodal_displacements, nodal_velocities, nodal_accelerations, Es, Ms, damping_force, assembled_gravity_force):
        self.time_steps = time_steps
        self.nodal_accelerations = nodal_accelerations
        self.nodal_velocities = nodal_velocities
        self.nodal_displacements = nodal_displacements
        self.Es = Es
        self.Ms = Ms
        self.damping_forces = damping_force
        self.assembled_gravity_force = assembled_gravity_force


