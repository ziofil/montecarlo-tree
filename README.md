# montecarlo-tree

A simple implementation of a model-agnostic montecarlo tree search to use with an actor-critic RL agent. Just pass a handler that interfaces with a specific model. The handler must implement the following functions:
- `num_actions(state)`
- `new_state(action, state)`
- `value(state)` <--- must also be able to work on lists/arrays of states
- `policy(state)`
- `next_possible_states(state)`
- `is_solution(state)` <--- not all problems will have a clear solution, in that case just always return False

## Usage

```python
from MonteCarloTreeSearch import MCTS

tree = MCTS(my_handler, max_steps = 10, num_simlulations = 100)

actions = tree.search()
```

voila.
