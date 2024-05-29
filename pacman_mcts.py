import golog
from itertools import product
from examples.pacman_golog import env, initial_state
from utils.mcts import GologNode, Policy_Player_MCTS

def pacman_goal(state):
    return all(not state.fluents[f'dot_present({d})'].value for d in initial_state.symbols['dot'])

def capsule_goal(state):
    return not any(state.fluents[f'capsule_present({c})'].value for c in initial_state.symbols['capsule'])

def ghost_goal(state):
    return state.fluents['pacman_powered_up()'].value and not state.fluents['ghost_present((2, 2))'].value

def main():
    # Define the MCTS Solver
    def mcts_solve(game, iterations=1000):
        root_node = GologNode(game, None, False, game.state, None)
        for _ in range(iterations):
            root_node.explore()
        next_tree, next_action, next_args = Policy_Player_MCTS(root_node)
        return next_action, next_args

    # Execute the solution found by MCTS
    def execute_mcts_solution(game, goal_check, iterations=1000):
        while not goal_check(game.state):
            best_action_index, best_action_args = mcts_solve(game, iterations)
            game.step((best_action_index, best_action_args))
            game.render()

    # Define procedural logic
    def find_and_eat_capsule(game):
        game.goal = capsule_goal
        execute_mcts_solution(game, capsule_goal)

    def find_and_eat_ghost(game):
        game.goal = ghost_goal
        execute_mcts_solution(game, ghost_goal)

    def find_and_eat_food(game):
        game.goal = pacman_goal
        execute_mcts_solution(game, pacman_goal)

    # Initialize the game
    env.reset()
    find_and_eat_capsule(env)
    find_and_eat_ghost(env)
    find_and_eat_food(env)
    env.close()

# Execute the main procedure
if __name__ == "__main__":
    main()