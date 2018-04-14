# Copyright 2018 Filippo Miatto
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


class MCTS():
    def __init__(self, handler, max_steps = 10, num_simulations = 100):
        """
        This MCTS is model-agnostic.
        The handler is responsible for interfacing with the Agent and it must implement the following functions:
        - num_actions(state)
        - new_state(action, state)
        - value(state) <--- must also be able to work on lists/arrays of states
        - policy(state)
        - next_possible_states(state)
        - is_solution(state) <--- not all problems will have a clear solution, in that case just return False

        It must also have the property handler.root, which is the state at the root of the tree.

        The tree is a dictionary, where the keys are the tuple of actions to get to that node and the value is a dictionary with the following keys:
        - state = state of the current node
        - visits = number of visits to the next nodes
        - values = value of the next nodes
        - policy = probability distribution over the next actions

        visits, values and policy must be numpy arrays.

        So each node contains information about the next states (the number of times they've been evaluated, their value, what the policy says),
        plus information about the current state.
        """
        self.handler = handler # object that acts, predicts, evaluates etc...
        self.tree = {}
        self.max_steps = max_steps
        self.num_simulations = num_simulations



    def _down_one_step(self, key):
        """
        This internal function extends the tree by going one step down. It is used only by the function search.
        The way it works is by picking an action by combining the value of the next steps and the policy (the variable choice).
        Then if the node already exists it just skips to it, or if it doesn't exist it creates it.
        As the policy is a probability but the value is not, there is a factor (below arbitrarily set to 10) to make the policy initially count as much as the value.
        Note that as the number of visits grows, the value counts more and more than the policy (becase the prob of picking an action goes like value + policy/visits).
        """
        node = {}
        try:
            node = self.tree[key]
            choice = node['values'] + 10*node['policy']/node['visits'] # factor of 10 is to make the policy have the same weight as the value.
            action = np.random.choice(self.handler.num_actions(node['state']), p=choice/sum(choice))
            #action = np.argmax(choice) # greedy (use after training is over)
            key = tuple([*key, action])
            try:
                # if the new node already exists, we're done.
                node = self.tree[key]
                return key, node['stop']
            except KeyError:
                state = self.handler.new_state(action, node['state']) # TODO: don't recompute this state?
        except KeyError:
            # we are at the root of the tree
            key = ()
            state = self.handler.root

        # if the new node doesn't exist we need to create it
        visits = np.ones([self.handler.num_actions(state)])
        values = self.handler.value(self.handler.next_possible_states(state)) # here handler.value needs to operate on a list/array of states
        policy = self.handler.policy(state)
        stop = self.handler.is_solution(state)
        self.tree.update({key:{'state':state, 'visits':visits, 'values':values, 'policy':policy, 'stop':stop}})

        return key, stop



    def _all_the_way_up(self, bottom_key):
        """
        This is the "learning' part. When going down the tree we just explored and stored information, now we use it.
        We start from the bottom and we go back up to the root.
        At each step we update the 'visits' and 'values' entries of the current node's dictionary to be (respectively) the total number of
        nodes that were explored below, and their average value.
        We do so by using two buffers, one for the total number of visits and one for the sum of all the values.
        """
        values_buffer = 0
        visits_buffer = 0
        key = bottom_key

        while True:
            node = self.tree[key]

            # --- update visits ---
            try: # fails at the bottom node (where we start our ascent), because we haven't specified an action
                node['visits'][action] += visits_buffer
            except NameError:
                pass
            visits_buffer = sum(node['visits']) # total visits of the node

            # --- update values ---
            try: # fails at the botton node
                node['values'][action] += values_buffer/node['visits'][action] # update value with average value of all lower nodes
            except NameError:
                pass
            values_buffer = node['values']@node['visits'] # this is an inner product (new total value of all nodes below the current one)

            self.tree[key] = node # update node

            try:
                key = list(key)
                action = key.pop() # go up the tree one step
                key = tuple(key)
            except IndexError:
                # when we hit the root, which has key = ()
                break



    def search(self, max_steps = None, num_simulations = None):
        """
        This is the only public function. Just call the search and it will return the vector of suggested probabilities for the next action.
        The way it works is just by going down and up the tree several times (self.num_simulations) to build a tree.
        Then the probabilities depend on the number of visits of the first nodes after the root.
        In the AlphaGo Zero paper they include a "temperature" parameter tau to further flatten/squeeze the final distribution, which I included for completeness.
        """
        # set specific parameters or pick the global ones from __init__
        _max_steps = max_steps or self.max_steps
        _num_simulations = num_simulations or self.num_simulations
        self.tree = {} # reset the tree

        for _ in range(_num_simulations):
            key = ()
            step = 0
            while True:
                key, stop = self._down_one_step(key)
                if stop:
                    # update visits with the number of possible nodes below (maybe it's too much? not sure)
                    self.tree[key]['visits'] += self.handler.num_actions(self.tree[key]['state'])**(_max_steps - step)
                    break
                if step == _max_steps:
                    break
                step += 1

            self._all_the_way_up(key)

        first_visits = self.tree[()]['visits']

        tau = 1 # "temperature parameter", see paper. The larger tau, the more we encourage exploration
        unnorm_probs = first_visits**(1/tau)

        # return prob distribution for picking the next action
        return unnorm_probs/np.sum(unnorm_probs)
