import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from examples.blocksworld_golog import env  # Assuming this is the environment you've set up

# Check if the environment follows the Gym API
check_env(env)

# Create the A2C model
model = A2C('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)

# Save the model
model.save("a2c_golog_blocksworld")

# Load the model
model = A2C.load("a2c_golog_blocksworld")


# Test the model
obs, _= env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)

    obs, rewards, _, done, info = env.step(action)
    print(f"Step {i}: Executing action: {action}")
    print(f"Reward: {rewards}")
    env.render()
    if done:
        print("Game over!")
        print(f"Total rewards: {rewards}")
        obs = env.reset()
        break
