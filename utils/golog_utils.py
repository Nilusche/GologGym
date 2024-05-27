class GologFluent:
    def __init__(self, domain, value):
        self.domain = domain
        self.value = value

    def set_value(self, value):
        if value in self.domain:
            self.value = value

    def __repr__(self):
        return str(self.value)

class GologState:
    def __init__(self):
        self.symbols = {}
        self.actions = []
        self.fluents = {}

    def add_symbol(self, symbol, domain):
        self.symbols[symbol] = domain
    
    def add_fluent(self, fluent, domain, initial_value):
        self.fluents[fluent] = GologFluent(domain, initial_value)

    def add_action(self, action):
        self.actions.append(action)
    
    def execute_action(self, action_index, *args):
        action = self.actions[action_index]
        if action.precondition(self, *args):
            action.effect(self, *args)
            return True
        return False

    def __hash__(self):
        return hash(frozenset((fluent, fl.value) for fluent, fl in self.fluents.items()))

    def __eq__(self, other):
        return all(self.fluents[fluent].value == other.fluents[fluent].value for fluent in self.fluents)

class GologAction:
    def __init__(self, name, precondition, effect, arg_domains):
        self.name = name
        self.precondition = precondition
        self.effect = effect
        self.arg_domains = arg_domains

    def generate_valid_args(self, state):
        from itertools import product
        domains = [state.symbols[domain] for domain in self.arg_domains]
        return list(product(*domains))
