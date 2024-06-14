from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import golog
from examples.pacman_golog import env

# Check if the environment follows the OpenAI Gym interface
#check_env(env, warn=True)

def filter_legal_actions(env, action_space):
    legal_actions = []
    for action_index in range(action_space.n):
        action, args = env.action_arg_combinations[action_index]
        if env.state.actions[action].precondition(env.state, *args):
            legal_actions.append(action_index)
    return legal_actions

def custom_step(env, model, obs):
    legal_actions = filter_legal_actions(env, env.action_space)
    action, _ = model.predict(obs)
    while action not in legal_actions:
        action, _ = model.predict(obs)
    return action


#Create the PPO agent
model = PPO('MlpPolicy', env, ent_coef=0.01, learning_rate=0.001, verbose=1)

# # Train the agent
model.learn(total_timesteps=300000, progress_bar=True)

# Save the model
model.save("ppo_pacman")

# # Load the model for further use
model = PPO.load("ppo_pacman")

# # Test the trained agent
obs, info = env.reset()


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
