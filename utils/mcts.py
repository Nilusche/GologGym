# mcts.py
from math import sqrt, log
from copy import deepcopy
import random

class GologNode:
    def __init__(self, game, parent, done, observation, action_index):
        self.child = None
        self.T = 0
        self.N = 0
        self.game = game
        self.observation = observation
        self.done = done
        self.parent = parent
        self.action_index = action_index

    def get_UCB_score(self, c=1.0):
        if self.N == 0:
            return float('inf')
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
        
        return self.T / self.N + c * sqrt(log(top_node.N) / self.N)

    def detach_parent(self):
        del self.parent
        self.parent = None

    def create_child(self):
        if self.done:
            return

        actions = list(range(self.game.action_space.n))
        games = [deepcopy(self.game) for _ in actions]
        
        child = {}
        for action, game in zip(actions, games):
            observation, reward, _, done, _ = game.step(action)
            child[action] = GologNode(game, self, done, observation, action)
        self.child = child

    def explore(self):
        current = self
        while current.child:
            child = current.child
            max_UCB = max(c.get_UCB_score() for c in child.values())
            actions = [action for action, node in child.items() if node.get_UCB_score() == max_UCB]
            if not actions:
                print("Error: zero length ", max_UCB)
            action = random.choice(actions)
            current = child[action]

        if current.N < 1:
            current.T += current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(list(current.child.values()))
            current.T += current.rollout()

        current.N += 1

        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T += current.T

    def rollout(self):
        if self.done:
            return 0

        v = 0
        done = False
        new_game = deepcopy(self.game)
        rollout_steps = 0

        while not done:  # Prevent infinite loops
            action_index = new_game.action_space.sample()
            observation, reward, _, done, _ = new_game.step(action_index)
            v += reward
            rollout_steps += 1
            if done:
                new_game.reset()
                new_game.close()
                break
        return v

    def next(self):
        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')
        
        child = self.child

        max_N = max(node.N for node in child.values())
        max_children = [c for a, c in child.items() if c.N == max_N]

        if not max_children:
            print("Error: zero length ", max_N)

        max_child = random.choice(max_children)

        return max_child, max_child.action_index


MCTS_POLICY_EXPLORE = 100  # Number of MCTS iterations

def Policy_Player_MCTS(mytree):
    for _ in range(MCTS_POLICY_EXPLORE):
        mytree.explore()
    
    next_tree, next_action = mytree.next()

    next_tree.detach_parent()
    return next_tree, next_action
