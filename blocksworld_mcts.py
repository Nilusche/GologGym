from copy import deepcopy
import golog
from examples.blocksworld_golog import env
from utils.mcts import GologNode, Policy_Player_MCTS

def main():
    reward_e = 0 
    observation = env.reset()
    done = False
    new_game = deepcopy(env)
    mytree = GologNode(new_game, None, False, observation, 0)
    step_counter = 0

    while not done:
        print(f"Step {step_counter}: Starting MCTS")
        mytree, action_index = Policy_Player_MCTS(mytree)
        observation, reward, done, _ = env.step(action_index)
        reward_e += reward
        action = env.action_arg_combinations[action_index]
        print(f"Step {step_counter}: Executing action: {env.state.actions[action[0]].name} with args {action[1:]}")
        env.render()

        if done:
            print('reward_e ' + str(reward_e))
            print("Game over!")
            break
        
        step_counter += 1

if __name__ == "__main__":
    main()
