

class DiffEq:
    """
    A representation of time dependent differential equations.
    """

    def __init__(self, independent_vars, conditions):
        self.independent_vars = independent_vars
        self.conditions = conditions

    def get_independent_vars(self):
        return self.independent_vars

    def get_conditions(self):
        return self.conditions
