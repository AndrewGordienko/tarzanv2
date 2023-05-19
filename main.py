from random import randint
from env5 import Simulation

class Agent():
    def choose_action(self, observation, direction):
        observation += [direction] * 17
        return randint(-10, 10)

agent = Agent()
direction = randint(0, 1)
simulation = Simulation()
simulation.run_through(agent, direction)

print(simulation.score)
