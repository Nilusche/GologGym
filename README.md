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
**Flexible Environment**: Easily define and manipulate Golog programs within a Gym-compatible framework. <br>
**Extensible**: Add new Golog operators and predicates to suit the needs of the specific domain. <br>

# Installation
Run `pip install -e golog` to register the golog environment <br>
Use the classes `GologAction`, `GologState` from `utils.golog_utils` to define golog state and actions for the program initiation <br>
Use `utils.mcts` to utilize custom mcts implementation <br>

# Folders


# Start
## Creating Your Own Golog Environment / Extending the GologEnv
To create a custom Golog environment, follow these steps:
1. **Define Your Golog Program**: Create a Golog program that specifies the behavior and logic of your environment.
2. **Implement the Environment**: Instantiate the Golog Environment with the GologState and GologActions
3. **Register the Environment**: Register the environment using the `register` function

# Examples
Check out the examples folder for various sample Golog programs and environments. These examples demonstrate how to create and interact with different Golog-based environments.
* **Blocksworld Example**: Environment Implementation can be found in blocksworld_golog.py 
* **Pacman Example**: Environment Implementation can be found in pacman_golog.py



