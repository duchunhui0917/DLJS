import random
import numpy as np


class Tetris:
    @staticmethod
    def calculate_alignment(valid_pairs):
        machine_features = []
        task_features = []
        for index, pair in enumerate(valid_pairs):
            machine = pair[0]
            task = pair[1]
            machine_features.append([machine.cpu, machine.memory])
            task_features.append([task.cpu, task.memory])
        return np.argmax(np.sum(np.array(machine_features) * np.array(task_features), axis=1), axis=0)

    def __call__(self, machines, jobs, sim):
        valid_pairs = []
        for machine in machines:
            for job in jobs:
                tasks = job.running_tasks
                for task in tasks:
                    if machine.fit(task):
                        valid_pairs.append((machine, task))
        if len(valid_pairs) == 0:
            return None, None
        pair_index = Tetris.calculate_alignment(valid_pairs)
        pair = valid_pairs[pair_index]
        return pair[0], pair[1]


class FirstFit:
    def __call__(self, machines, jobs, sim):
        machine_task_pairs = []
        for machine in machines:
            for job in jobs:
                tasks = job.running_tasks
                for task in tasks:
                    if machine.fit(task):
                        machine_task_pairs.append((machine, task))
        if len(machine_task_pairs) == 0:
            candidate_machine, candidate_task = None, None
        else:
            candidate_machine, candidate_task = machine_task_pairs[0]
        return candidate_machine, candidate_task


class Random:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def __call__(self, machines, jobs, sim):
        candidate_task = None
        candidate_machine = None
        all_candidates = []

        for machine in machines:
            for job in jobs:
                tasks = job.running_tasks
                for task in tasks:
                    if machine.fit(task):
                        all_candidates.append((machine, task))
                        if np.random.rand() > self.threshold:
                            candidate_machine = machine
                            candidate_task = task
                            break
        if len(all_candidates) == 0:
            return None, None
        if candidate_task is None:
            pair_index = np.random.randint(0, len(all_candidates))
            return all_candidates[pair_index]
        else:
            return candidate_machine, candidate_task
