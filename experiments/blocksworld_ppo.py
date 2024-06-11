from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import golog
from examples.blocksworld_golog import env

# Check if the environment follows the OpenAI Gym interface
#check_env(env, warn=True)

# Create the PPO agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=3000)

# Save the model
model.save("ppo_blocks_world")

# Load the model for further use
model = PPO.load("ppo_blocks_world")

# Test the trained agent
obs, info = env.reset()
rewards = 0 
done = False
i = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, _,  done, info = env.step(action)
    rewards += rewards
    print(f"Step {i}: Executing action: {action}(args: {env.action_arg_combinations[action][1:]})")
    env.render()
    i += 1
    if done:
        print('rewards ' + str(rewards))
        print("Game over!")
        break
