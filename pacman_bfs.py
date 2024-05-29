import golog
from itertools import product
import time
from examples.pacman_golog import env, pacman_goal, initial_state
import gym
import pygame
from collections import deque
import copy




def main():
    def bfs_solve(current_state, goal_check):
        queue = deque([(copy.deepcopy(current_state), [])])  # Queue of (state, action_sequence)
        visited = set()

        while queue:
            current_state, action_sequence = queue.popleft()
            if goal_check(current_state):
                return action_sequence

            for action_index, action in enumerate(current_state.actions):
                for args in action.generate_valid_args(current_state):
                    if action.precondition(current_state, *args):
                        next_state = copy.deepcopy(current_state)
                        action.effect(next_state, *args)
                        state_hash = hash(next_state)
                        if state_hash not in visited:
                            visited.add(state_hash)
                            queue.append((next_state, action_sequence + [(action_index, args)]))

        return None

    # Execute the solution found by BFS
    def execute_solution(game, solution):
        for step, (action_index, args) in enumerate(solution):
            print(f"Step {step + 1}: {game.state.actions[action_index].name}{args}")
            game.step((action_index, args))
            game.render()

    # Define procedural logic
    def find_and_eat_capsule(game):
        def capsule_goal(state):
            return not any(state.fluents[f'capsule_present({c})'].value for c in initial_state.symbols['capsule'])
        solution = bfs_solve(game.state, capsule_goal)
        if solution:
            print("Capsule solution found:")
            execute_solution(game, solution)
        else:
            print("No capsule solution found")

    def find_and_eat_ghost(game):
        def ghost_goal(state):
            return state.fluents['pacman_powered_up()'].value and not state.fluents['ghost_present((2, 2))'].value
        solution = bfs_solve(game.state, ghost_goal)
        if solution:
            print("Ghost solution found:")
            execute_solution(game, solution)
        else:
            print("No ghost solution found")

    def find_and_eat_food(game):
        solution = bfs_solve(game.state, pacman_goal)
        if solution:
            print("Food solution found:")
            execute_solution(game, solution)
        else:
            print("No food solution found")

    # Initialize the game
    env.reset()
    find_and_eat_capsule(env)
    find_and_eat_ghost(env)
    find_and_eat_food(env)
    env.close()



if __name__ == "__main__":
    main()