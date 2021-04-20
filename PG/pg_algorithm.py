from torch.nn import functional as F
import torch
from torch.distributions import Categorical
import numpy as np


eps = np.finfo(np.float64).eps.item()


class PGAlgorithm:
    def __init__(self, agent):
        self.agent = agent
        self.trajectory = []

    def __call__(self, machines, jobs, sim):
        machine_task_pairs = []
        for machine in machines:
            for job in jobs:
                tasks = job.running_tasks
                for task in tasks:
                    if machine.fit(task):
                        machine_task_pairs.append((machine, task))

        if len(machine_task_pairs) == 0:
            self.trajectory.append((None, None, -1, sim.now))
            return None, None
        else:
            features = self.extract_feature(machine_task_pairs)
            logits = F.softmax(self.agent.actor(features), dim=1)
            pair_index = Categorical(logits).sample().item()
            machine, task = machine_task_pairs[pair_index]
            self.trajectory.append((features, pair_index, 0, sim.now))
            return machine, task

    @staticmethod
    def extract_feature(state):
        features = []
        for s in state:
            machine, task = s
            feature = [machine.cpu, machine.memory, task.cpu, task.memory, task.duration,
                       task.instances_num - task.running_instance_pointer]
            features.append(feature)
        y = (np.array(features) - np.array([0, 0, 0.65, 0.009, 74.0, 80.3])) / np.array(
            [64, 1, 0.23, 0.005, 108.0, 643.5])
        return torch.from_numpy(y).float()

