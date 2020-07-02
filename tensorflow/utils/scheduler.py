class StepScheduler(object):
    def __init__(self, name, base_value, decay_rate, decay_step, max_steps, clip_min=0):
        self.name = name
        self.clip_min = clip_min
        self.cur_step = 0
        self.values = [base_value * decay_rate ** (i // decay_step) for i in range(max_steps)]

    def reset(self):
        self.cur_step = 0

    def step(self):
        cur_value = max(self.values[self.cur_step], self.clip_min)
        self.cur_step += 1
        return cur_value
