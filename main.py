import multiprocessing
import time
from random import randint
from env5 import Simulation

class Agent():
    def choose_action(self, observation, direction):
        observation += [direction] * 17
        return randint(-10, 10)

def simulate_environment(seed):
    agent = Agent()
    direction = randint(0, 1)
    simulation = Simulation()
    simulation.run_through(agent, direction)
    return simulation.score

if __name__ == '__main__':
    num_envs = 4000  # Number of environments to run in parallel
    pool = multiprocessing.Pool(processes=num_envs)
    
    start_time = time.time()  # Start timer
    scores = pool.map(simulate_environment, range(num_envs))
    end_time = time.time()  # End timer
    
    print("Number of Enviornments {} Total execution time: {} seconds".format(num_envs, end_time - start_time))
    print(scores)
