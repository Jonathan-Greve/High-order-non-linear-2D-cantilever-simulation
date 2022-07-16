# Material properties class with young's modulus, poisson ratio, and density
# Has a function for computing lambda and mu
class MaterialProperties:
    def __init__(self, youngs_modulus, poisson_ratio, density, damping_coefficient):
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.damping_coefficient = damping_coefficient

    def get_lambda_and_mu(self):
        lambda_ = self.youngs_modulus * self.poisson_ratio / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        mu = self.youngs_modulus / (2 * (1 + self.poisson_ratio))
        return lambda_, mu


# Class for querying for a material properties class
# contains MaterialProperties for:
#   - Steel
#   - Wood
#   - Concrete
#   - Aluminium
#   - Glass
#   - Rubber
#   - Plastic
#   - Gold
#   - Silver
#   - Bronze
#   - Iron
#   - Titanium
class MaterialPropertiesQuery:
    def __init__(self):
        self.material_properties = dict()
        self.material_properties["Steel"] = MaterialProperties(
            youngs_modulus=69.0e9,
            poisson_ratio=0.3,
            density=7.8e3,
            damping_coefficient=0.01)
        self.material_properties["Wood"] = MaterialProperties(
            youngs_modulus=1.0e11,
            poisson_ratio=0.3,
            density=1.0e3,
            damping_coefficient=0.1)
        self.material_properties["Concrete"] = MaterialProperties(
            youngs_modulus=1.0e11,
            poisson_ratio=0.3,
            density=2.5e3,
            damping_coefficient=0.01)
        self.material_properties["Aluminium"] = MaterialProperties(
            youngs_modulus=2.0e11,
            poisson_ratio=0.3,
            density=2.7e3,
            damping_coefficient=0.01)
        self.material_properties["Glass"] = MaterialProperties(
            youngs_modulus=2.0e11,
            poisson_ratio=0.3,
            density=2.2e3,
            damping_coefficient=0.01)
        self.material_properties["Rubber"] = MaterialProperties(
            youngs_modulus=0.01e09,
            poisson_ratio=0.48,
            density=1050,
            damping_coefficient=0.10)
        self.material_properties["Plastic"] = MaterialProperties(
            youngs_modulus=0.01e9,
            poisson_ratio=0.48,
            density=1.0e3,
            damping_coefficient = 0.001)
        self.material_properties["Gold"] = MaterialProperties(
            youngs_modulus=2.0e11,
            poisson_ratio=0.3,
            density=1.0e3,
            damping_coefficient=0.001)
        self.material_properties["Test 1"] = MaterialProperties(
            youngs_modulus=10.0e5,
            poisson_ratio=0.3,
            density=1000,
            damping_coefficient=0.00)

    def get_material_properties(self, material_name):
        if material_name in self.material_properties:
            return self.material_properties[material_name]
        else:
            return None
