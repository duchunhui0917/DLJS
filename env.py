from myclass import Job, Task, Machine
import simpy

MACHINE_NUMBER = 5


class Env:
    def __init__(self, jobs_config):
        self.sim_env = simpy.Environment()
        self.job_configs = jobs_config
        self.jobs = []
        self.machines = []
        # homogeneous
        for i in range(MACHINE_NUMBER):
            self.machines.append(Machine(i, 64, 1))
        # # heterogeneous
        # self.machines.append((Machine(1, 64, 1)))
        # self.machines.append(Machine(2, 32, 0.5))
        # self.machines.append(Machine(3, 128, 1))
        # self.machines.append(Machine(4, 64, 2))
        # self.machines.append(Machine(5, 32, 1))

        self.broker_finished = False

    def broker(self, sim):
        for job_config in self.job_configs:
            assert job_config.submit_time >= sim.now
            yield sim.timeout(job_config.submit_time - sim.now)
            job = Job(job_config)
            self.jobs.append(job)
        self.broker_finished = True

    def schedule(self, sim, agent):
        while not self.is_finished:
            while True:
                machine, task = agent(self.machines, self.jobs, sim)
                if machine is None and task is None:
                    break
                else:
                    machine.work(task, sim)
            yield sim.timeout(1)

    @property
    def is_finished(self):
        flag = self.broker_finished
        for job in self.jobs:
            flag = flag and job.is_finished
        return flag
