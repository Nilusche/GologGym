import gymnasium as gym
from examples.blocksworld_golog import env
from utils.mcts import GologNode, Policy_Player_MCTS

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
        env.render()

        if done:
            print(f"Selected action: {best_action}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            env.render()
            break

        print(f"Selected action: {best_action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

    
if __name__ == "__main__":
    main()
