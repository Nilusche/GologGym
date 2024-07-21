import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from examples.blocksworld_golog import env
import numpy as np
# Define the mask function
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

# Wrap the environment with the ActionMasker
env = ActionMasker(env, mask_fn)

# # Create the MaskablePPO model
model = MaskablePPO('MlpPolicy', env, verbose=1, tensorboard_log="./logs/Policy/maskable_ppo_golog_blocksworld_tensorboard/")

# Train the model
model.learn(total_timesteps=50000, progress_bar=True)

# Save the model
# model.save("maskable_ppo_golog_pacman")

# # Load the model
# model = MaskablePPO.load("maskable_ppo_golog_pacman", env=env)

# Test the trained agent
obs, _ = env.reset()
rewards = 0
done = False
i = 0
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, _, info = env.step(action)
    rewards += reward
    print(f"Step {i}: Executing action: {action} with reward {reward}")
    env.render()
    i += 1
    if done:
        print('Total rewards:', rewards)
        print("Game over!")
        break
