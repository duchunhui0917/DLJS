class Job:
    def __init__(self, job_config):
        self.job_id = job_config.job_id
        self.submit_time = job_config.submit_time
        self.tasks = {}
        for task_config in job_config.task_configs:
            task_id = task_config.task_id
            self.tasks[task_id] = Task(task_config)

    @property
    def running_tasks(self):
        tasks = self.tasks.values()
        return [task for task in tasks if task.is_adding]

    @property
    def is_finished(self):
        tasks = self.tasks.values()
        for task in tasks:
            if not task.is_finished:
                return False
        return True


class Task:
    def __init__(self, task_config):
        self.task_id = task_config.task_id
        self.submit_time = task_config.submit_time
        self.instances_num = task_config.instances_num
        self.cpu = task_config.cpu
        self.memory = task_config.memory
        self.disk = task_config.disk
        self.duration = task_config.duration

        self.running_instance_pointer = 0
        self.finished_instance_pointer = 0

    @property
    def is_finished(self):
        return self.running_instance_pointer == self.finished_instance_pointer == self.instances_num

    @property
    def is_adding(self):
        return self.running_instance_pointer < self.instances_num


class JobConfig(object):
    def __init__(self, job_id, submit_time, task_configs):
        self.job_id = job_id
        self.submit_time = submit_time
        self.task_configs = task_configs


class TaskConfig(object):
    def __init__(self, task_id, submit_time, instances_num, cpu, memory, disk, duration, parent_indices=None):
        self.task_id = task_id
        self.submit_time = submit_time
        self.instances_num = instances_num
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.duration = duration
        self.parent_indices = parent_indices


class Machine:
    def __init__(self, i, cpu_capacity, memory_capacity):
        self.id = i
        self.cpu_capacity = self.cpu = cpu_capacity
        self.memory_capacity = self.memory = memory_capacity

    def fit(self, instance):
        return self.cpu > instance.cpu and self.memory > instance.memory

    def work(self, task, sim):
        self.allocate(task)
        sim.process(self.free(task, sim))

    def allocate(self, task):
        self.cpu -= task.cpu
        self.memory -= task.memory
        task.running_instance_pointer += 1

    def free(self, task, sim):
        yield sim.timeout(task.duration)
        self.cpu += task.cpu
        self.memory += task.memory
        task.finished_instance_pointer += 1
