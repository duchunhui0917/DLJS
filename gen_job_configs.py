from operator import attrgetter
import pandas as pd
import numpy as np
import random

from myclass import JobConfig, TaskConfig


def statistic(ret):
    the_first_job_config = ret[0]
    submit_time_base = the_first_job_config.submit_time

    tasks_number = 0
    submit_time_difs = []
    task_instances_numbers = []
    task_instances_durations = []
    task_instances_cpu = []
    task_instances_memory = []
    for job_config in ret:
        job_config.submit_time -= submit_time_base
        if len(submit_time_difs) == 0:
            submit_time_difs.append(0)
        else:
            submit_time_difs.append(job_config.submit_time - submit_time_difs[-1])
        tasks_number += len(job_config.task_configs)
        for task_config in job_config.task_configs:
            task_instances_numbers.append(task_config.instances_num)
            task_instances_durations.extend([task_config.duration] * int(task_config.instances_num))
            task_instances_cpu.extend([task_config.cpu] * int(task_config.instances_num))
            task_instances_memory.extend([task_config.memory] * int(task_config.instances_num))

    print('Jobs number: ', len(ret))
    print('Tasks number:', tasks_number)

    print('Job submit time difs mean ', np.mean(submit_time_difs))
    print('Job submit time difs std: ', np.std(submit_time_difs))

    print('Task instances number mean: ', np.mean(task_instances_numbers))
    print('Task instances number std', np.std(task_instances_numbers))

    print('Task instances cpu mean: ', np.mean(task_instances_cpu))
    print('Task instances cpu std: ', np.std(task_instances_cpu))

    print('Task instances memory mean: ', np.mean(task_instances_memory))
    print('Task instances memory std: ', np.std(task_instances_memory))

    print('Task instances duration mean: ', np.mean(task_instances_durations))
    print('Task instances duration std: ', np.std(task_instances_durations))


class CSVJobs:
    def __init__(self, filename):
        self.filename = filename
        df = pd.read_csv(self.filename)

        df.task_id = df.task_id.astype(dtype=int)
        df.job_id = df.job_id.astype(dtype=int)
        df.instances_num = df.instances_num.astype(dtype=int)

        job_task_map = {}
        job_submit_time_map = {}
        for i in range(len(df)):
            series = df.iloc[i]
            job_id = series.job_id
            task_id = series.task_id

            cpu = series.cpu
            memory = series.memory
            disk = series.disk
            duration = series.duration
            submit_time = series.submit_time
            instances_num = series.instances_num

            task_configs = job_task_map.setdefault(job_id, [])
            task_configs.append(TaskConfig(task_id, submit_time, instances_num, cpu, memory, disk, duration))
            job_submit_time_map[job_id] = submit_time

        job_configs = []
        for job_id, task_configs in job_task_map.items():
            job_configs.append(JobConfig(job_id, job_submit_time_map[job_id], task_configs))
        job_configs.sort(key=attrgetter('submit_time'))

        self.job_configs = job_configs

    def generate(self, offset, number):
        number = number if offset + number < len(self.job_configs) else len(self.job_configs) - offset
        ret = self.job_configs[offset: offset + number]
        statistic(ret)
        return ret


class RandomJobs:
    def __init__(self, jobs_num, tasks_mean, tasks_std, instances_mean, instances_std, submit_time_mean,
                 submit_time_std, cpu_mean, cpu_std, memory_mean, memory_std, duration_mean, duration_std):
        self.jobs_num = jobs_num
        self.tasks_mean = tasks_mean
        self.tasks_std = tasks_std
        self.instances_mean = instances_mean
        self.instances_std = instances_std
        self.submit_time_mean = submit_time_mean
        self.submit_time_std = submit_time_std
        self.cpu_mean = cpu_mean
        self.cpu_std = cpu_std
        self.memory_mean = memory_mean
        self.memory_std = memory_std
        self.duration_mean = duration_mean
        self.duration_std = duration_std

    def generate(self):
        job_configs = []
        submit_time = 0
        for job_id in range(self.jobs_num):
            submit_time += np.random.random() * self.submit_time_std + self.submit_time_mean
            task_configs = []
            for task_id in range(random.randint(self.tasks_mean - self.tasks_std, self.tasks_mean + self.tasks_std)):
                instances_num = random.randint(self.instances_mean - self.instances_std,
                                               self.instances_mean + self.instances_std)
                cpu = random.random() * self.cpu_std + self.cpu_mean
                memory = random.random() * self.memory_std + self.memory_mean
                disk = 0
                duration = random.random() * self.duration_std + self.duration_mean
                task_configs.append(TaskConfig(task_id, submit_time, instances_num, cpu, memory, disk, duration))
            job_configs.append(JobConfig(job_id, submit_time, task_configs))
        statistic(job_configs)
        return job_configs
