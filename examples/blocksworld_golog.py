import golog
import gym
from utils.golog_utils import GologState, GologAction


def stack_precondition(state, x, y):
    return x != y and x != 'table' and state.fluents[f'loc({x})'].value != y and not any(state.fluents[f'loc({z})'].value == x for z in state.symbols['block'])

def stack_effect(state, x, y):
    state.fluents[f'loc({x})'].set_value(y)

def blocksworld_goal(state):
    return state.fluents['loc(a)'].value == 'table' and state.fluents['loc(b)'].value == 'a' and state.fluents['loc(c)'].value == 'b'

initial_state = GologState()
initial_state.add_symbol('block', ['a', 'b', 'c'])
initial_state.add_symbol('location', ['a', 'b', 'c', 'table'])
initial_state.add_fluent('loc(a)', ['a', 'b', 'c', 'table'], 'c')
initial_state.add_fluent('loc(b)', ['a', 'b', 'c', 'table'], 'table')
initial_state.add_fluent('loc(c)', ['a', 'b', 'c', 'table'], 'b')

stack_action = GologAction('stack', stack_precondition, stack_effect, [initial_state.symbols['block'], initial_state.symbols['location']])
initial_state.add_action(stack_action)

actions = [
    GologAction('stack', stack_precondition, stack_effect, ['block', 'location']),
]

env = gym.make('Golog-v0', initial_state=initial_state, goal_function=blocksworld_goal, actions=actions)
