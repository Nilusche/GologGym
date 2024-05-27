import golog
from collections import deque
import copy
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




def bfs_solve(env):
    initial_obs = env.reset()
    queue = deque([(copy.deepcopy(env.state), [])])  # Queue of (state, action_sequence)
    visited = set()
    
    while queue:
        current_state, action_sequence = queue.popleft()
        if blocksworld_goal(current_state):
            return action_sequence
        
        for action_index, action in enumerate(env.state.actions):
            for block in env.state.symbols['block']:
                for location in env.state.symbols['location']:
                    if action.precondition(current_state, block, location):
                        next_state = copy.deepcopy(current_state)
                        action.effect(next_state, block, location)
                        if next_state not in visited:
                            visited.add(next_state)
                            queue.append((next_state, action_sequence + [(action_index, block, location)]))
    
    return None

def main():
    solution = bfs_solve(env)
    if solution:
        print("Solution found:")
        for step, (action_index, block, location) in enumerate(solution):
            print(f"Step {step + 1}: {env.state.actions[action_index].name}({block}, {location})")
        
        # Apply the solution actions to the environment
        env.reset()
        for action_index, block, location in solution:
            env.step((action_index, [block, location]))
        
        # Render the final state
        env.render()
    else:
        print("No solution found")

if __name__ == "__main__":
    main()