import golog
from collections import deque
import copy
import gym
from examples.blocksworld_golog import env, blocksworld_goal



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