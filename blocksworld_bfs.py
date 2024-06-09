import golog
from collections import deque
import copy
import gym
from examples.blocksworld_golog import env, blocksworld_goal
import copy


def bfs_solve(env):
    initial_obs = env.reset()
    queue = deque([(copy.deepcopy(env.state), [])])  # Queue of (state, action_sequence)
    visited = set()
    
    while queue:
        current_state, action_sequence = queue.popleft()
        if blocksworld_goal(current_state):
            return action_sequence
        
        for action_index, action_args in enumerate(env.action_arg_combinations):
            action = env.state.actions[action_args[0]]
            args = action_args[1]
            if action.precondition(current_state, *args):
                next_state = copy.deepcopy(current_state)
                action.effect(next_state, *args)
                state_hash = hash(frozenset((fluent, fl.value) for fluent, fl in next_state.fluents.items()))
                if state_hash not in visited:
                    visited.add(state_hash)
                    queue.append((next_state, action_sequence + [(action_index, *args)]))
    
    return None

def main():
    solution = bfs_solve(env)
    if solution:
        print("Solution found:")
        for step, (action_index, *args) in enumerate(solution):
            action = env.state.actions[env.action_arg_combinations[action_index][0]]
            print(f"Step {step + 1}: {action.name}{tuple(args)}")
        
        # Apply the solution actions to the environment
        env.reset()
        for action_index, *args in solution:
            env.step(action_index)
        
        # Render the final state
        env.render()
    else:
        print("No solution found")

if __name__ == "__main__":
    main()