# GologGym
GologGym is an OpenAI Gym environment tailored for Golog programs. It provides a platform for testing, training, and benchmarking Golog-based decision-making algorithms within the Gym framework.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Folders](#folders)
4. [Start](#Start)
5. [Examples](#examples)

# Introduction
GologGym integrates the powerful decision-making capabilities of Golog with the versatile and widely-used OpenAI Gym environment. This allows developers and researchers to leverage Golog’s high-level programming constructs in the context of reinforcement learning and other AI research areas.

# Features
**Seamless Integration**: Combine the high-level action languages of Golog with the robust reinforcement learning environment of OpenAI Gym.
**Flexible Environment**: Easily define and manipulate Golog programs within a Gym-compatible framework. <br>
**Extensible**: Add new Golog operators and predicates to suit the needs of the specific domain. <br>

# Installation
Run `pip install -e golog` to register the golog environment <br>
Use the classes `GologAction`, `GologState` from `utils.golog_utils` to define golog state and actions for the program initiation <br>
Use `utils.mcts` to utilize custom mcts implementation <br>

# Folders
* `/examples`: Contains example environment for the Blocksworld and Pacman in Prolog Environment
* `files`: Contains the pseudo language files to define the Blocksworld and Pacman Environment
* `/golog`: Contains the Golog Environment implementation
* `/utils`: Contains utility functions for Golog Environment


# Start
## Define your Golog Program
To define your golog program as env follow these steps:
1. **Define your domain**: The domain includes all the objects, fluents (variables that describe the state of the world), and initial state of the environment
2. **Define your actions**: Actions are defined by their preconditions and effects. Preconditions are conditions that must be true for the action to be executed, and effects describe how the state changes after the action is executed.
3. **Create the Environment**: Now create the Golog environment using the initial state, actions and goal function.
Example:
```python
def stack_precondition(state, x, y):
    return x != y and x != 'table' and state.fluents[f'loc({x})'].value != y and not any(state.fluents[f'loc({z})'].value == x for z in state.symbols['block'])

def stack_effect(state, x, y):
    state.fluents[f'loc({x})'].set_value(y)

def blocksworld_goal(state):
    return state.fluents['loc(a)'].value == 'table' and state.fluents['loc(b)'].value == 'a' and state.fluents['loc(c)'].value == 'b'

initial_state = GologState()
#name, values
initial_state.add_symbol('block', ['a', 'b', 'c'])
initial_state.add_symbol('location', ['a', 'b', 'c', 'table'])
#name, possible values, initial val
initial_state.add_fluent('loc(a)', ['a', 'b', 'c', 'table'], 'c')
initial_state.add_fluent('loc(b)', ['a', 'b', 'c', 'table'], 'table')
initial_state.add_fluent('loc(c)', ['a', 'b', 'c', 'table'], 'b')

stack_action = GologAction('stack', stack_precondition, stack_effect, [initial_state.symbols['block'], initial_state.symbols['location']])
initial_state.add_action(stack_action)

#can be passed as array or created with the add_action() function.
actions = [
    GologAction('stack', stack_precondition, stack_effect, ['block', 'location']),
]
#Create the Environment
env = gym.make('Golog-v0', initial_state=initial_state, goal_function=blocksworld_goal, actions=actions)
```


## Creating Your Own Golog Environment / Extending the GologEnv
To create a custom Golog environment, follow these steps:
1. **Define Your Golog Program**: Create a Golog program that specifies the behavior and logic of your environment.
2. **Implement the Environment**: Instantiate the Golog Environment with the GologState and GologActions
3. **Register the Environment**: Register the environment using the `register` function
Example:
```python
from golog_gym.envs import GologGymEnv
from gym.envs.registration import register

class MyGologEnv(GologGymEnv):
    def __init__(self):
        super(MyGologEnv, self).__init__()

    def reset(self):
        # Your reset logic
        pass

    def step(self, action):
        # Your step logic
        pass

    def render(self, mode='human'):
        # Your render logic
        pass

register(
    id='MyGologEnv-v0',
    entry_point='my_module:MyGologEnv',
)
```


# Examples
Check out the examples folder for various sample Golog programs and environments. These examples demonstrate how to create and interact with different Golog-based environments.
* **Blocksworld Example**: Environment Implementation can be found in blocksworld_golog.py 
* **Pacman Example**: Environment Implementation can be found in pacman_golog.py



