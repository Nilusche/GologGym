import gymnasium as gym
from examples.blocksworld_golog import env
from utils.mcts import GologNode, Policy_Player_MCTS
import time
def main():
    # Initialize the environment
    done = False
    # Reset the environment to get the initial observation
    observation, _ = env.reset()

    # Initialize the root node of the MCTS
    root = GologNode(env, parent=None, done=False, observation=observation, action_index=None)

    while not done:
       
        # Run MCTS to get the best action
        root, best_action = Policy_Player_MCTS(root)

        # Apply the best action to the environment
        observation, reward, _, done, _ = env.step(best_action)

        # Render the environment to visualize the result
        #env.render()

        if done:
            # print(f"Selected action: {best_action}")
            # print(f"Reward: {reward}")
            # print(f"Done: {done}")
            # env.render()
            return reward
            break

        #print(f"Selected action: {best_action}")
        #print(f"Reward: {reward}")
        #print(f"Done: {done}")
    return reward

def execute():
    start = time.time()
    reward = main()
    end = time.time()
    #print milliseconds
    miliseconds = (end - start) * 1000
    return miliseconds, reward
    
if __name__ == "__main__":
    #time execution
    
    list_of_times = []
    list_of_rewards = []
    for i in range(50):
        milliseconds, reward = execute()
        list_of_times.append(milliseconds)
        list_of_rewards.append(reward)

    #print average time
    print(sum(list_of_times)/len(list_of_times))
    #print average reward
    print(sum(list_of_rewards)/len(list_of_rewards))
    print(list_of_times)
    print(list_of_rewards)
