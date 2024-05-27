from copy import deepcopy
import golog
import gym
from utils.golog_utils import GologState, GologAction
from utils.mcts import GologNode, Policy_Player_MCTS

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


game = gym.make('Golog-v0', initial_state=initial_state, goal_function=blocksworld_goal, actions=actions)

def main():
    reward_e = 0 
    observation = game.reset()
    done = False
    new_game = deepcopy(game)
    mytree = GologNode(new_game, None, False, observation, 0)
    step_counter = 0

    while not done:
        print(f"Step {step_counter}: Starting MCTS")
        mytree, action_index, args = Policy_Player_MCTS(mytree)
        observation, reward, done, _ = game.step((action_index, args))
        reward_e += reward
        print(f"Step {step_counter}: Executing action: {game.state.actions[action_index].name} with args {args}")
        game.render()

        if done:
            print('reward_e ' + str(reward_e))
            print("Game over!")
            break
        
        step_counter += 1


if __name__ == "__main__":
    main()