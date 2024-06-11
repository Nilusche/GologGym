from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import golog
from examples.pacman_golog import env
from itertools import product

# Check if the environment follows the OpenAI Gym interface
#check_env(env, warn=True)

def filter_legal_actions(env, action_space):
    legal_actions = []
    for action_index in range(action_space.nvec[0]):
        for args_combination in product(*(range(n) for n in action_space.nvec[1:])):
            action_combination = [action_index] + list(args_combination)
            action_obj = env.state.actions[action_index]
            arg_values = [env.state.symbols[domain][arg] for domain, arg in zip(action_obj.arg_domains, args_combination)]
            if action_obj.precondition(env.state, *arg_values):
                legal_actions.append(action_combination)
    return legal_actions



def custom_step(env, model, obs):
    legal_actions = filter_legal_actions(env, env.action_space)
    action, _ = model.predict(obs)
    action = action.tolist()  # Ensure action is in the same format as legal_actions
    while action not in legal_actions:
        action, _ = model.predict(obs)
        action = action.tolist()
    return action


# # Create the PPO agent
# model = PPO('MlpPolicy', env, ent_coef=0.1, learning_rate=0.0003, verbose=1)

# # # Train the agent
# model.learn(total_timesteps=100000, progress_bar=True)

# # Save the model
# model.save("ppo_pacman")

# # Load the model for further use
model = PPO.load("ppo_pacman")

# # Test the trained agent
obs, info = env.reset()


rewards = 0 
done = False
i = 0
while not done:
    action = custom_step(env, model, obs)
    obs, reward, _,  done, info = env.step(action)
    rewards += reward
    print(f"Step {i}: Executing action: {action})")
    print(f"Reward: {reward}")
    env.render()
    i += 1
    if done:
        print('rewards ' + str(rewards))
        print("Game over!")
        break
