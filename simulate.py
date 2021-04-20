import simpy
from env import Env
from gen_job_configs import CSVJobs, RandomJobs
from PG.pg_algorithm import PGAlgorithm
from PG.agent import Agent
from heuristic import FirstFit, Random, Tetris
import numpy as np
import torch
import time

n_input = 6
n_output = 1
ITERATION = 100
EPISODE = 12

np.random.seed(666)
torch.manual_seed(189)


def episode(env, algorithm):
    torch.manual_seed(int(time.time()))
    sim = simpy.Environment()
    sim.process(env.broker(sim))
    sim.process(env.schedule(sim, algorithm))
    sim.run()
    return sim.now


def run(csv_jobs=True):
    if csv_jobs:
        csv_jobs = CSVJobs('./jobs.csv')
        jobs_config = csv_jobs.generate(0, 10)
    else:
        random_jobs = RandomJobs(jobs_num=10, tasks_mean=9, tasks_std=7, instances_mean=45, instances_std=40,
                                 submit_time_mean=81, submit_time_std=53, cpu_mean=0.5, cpu_std=0.1, memory_mean=0.009,
                                 memory_std=0.003, duration_mean=75, duration_std=45)
        jobs_config = random_jobs.generate()
    makespan = episode(Env(jobs_config), Random())
    print('Random makespan:', makespan)
    makespan = episode(Env(jobs_config), FirstFit())
    print('FirstFit makespan:', makespan)
    makespan = episode(Env(jobs_config), Tetris())
    print('Tetris makespan:', makespan)

    agent = Agent(n_input, n_output)
    for i_iteration in range(ITERATION):
        print('******iteration%d******' % i_iteration)
        all_states = []
        all_actions = []
        all_rewards = []

        makespans = []
        trajectories = []

        start_time = time.time()
        for i_episode in range(EPISODE):
            env = Env(jobs_config)
            pg_algorithm = PGAlgorithm(agent)
            makespan = episode(env, pg_algorithm)
            makespans.append(makespan)
            trajectories.append(pg_algorithm.trajectory)
        print('%d episode spending:%.3f' % (EPISODE, time.time() - start_time))

        for trajectory in trajectories:
            states = []
            actions = []
            rewards = []
            for transition in trajectory:
                state, action, reward, _ = transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)

        makespan = sum(makespans) / len(makespans)

        all_qs, all_advantages = agent.estimate_return(all_rewards)

        start_time = time.time()
        agent.update_parameters(all_states, all_actions, all_advantages)
        print('Updating parameters spending:', time.time() - start_time)

        print('makespan:', makespans, 'average makespan:', makespan)


if __name__ == '__main__':
    run()
