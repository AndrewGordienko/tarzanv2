import multiprocessing
import time
from random import randint
from env5 import Simulation


def simulate_environment(seed):
    direction_moving = randint(0, 1)
    simulation = Simulation()
    simulation.run_through(direction_moving)
    return simulation.score

if __name__ == '__main__':
    num_envs = 1  # Number of environments to run in parallel
    pool = multiprocessing.Pool(processes=num_envs)
    
    start_time = time.time()  # Start timer
    scores = pool.map(simulate_environment, range(num_envs))
    end_time = time.time()  # End timer
    
    print("Number of Enviornments {} Total execution time: {} seconds".format(num_envs, end_time - start_time))
    print(sum(scores)/num_envs)
