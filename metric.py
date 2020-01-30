class Metric(object):
    def __init__(self, result):
        self.result = result
        self.n_task = self.result.shape[0]
        self.n_total = self.n_task * (self.n_task + 1) / 2.0

    def compute(self):
        pass


class AverageAccuracy(Metric):
    def __init__(self, result):
        super(AverageAccuracy, self).__init__(result)

    def compute(self):
        self.avg_acc = self.result[self.n_task - 1].sum() / self.n_task

        return self.avg_acc


class TotalAccuracy(Metric):
    def __init__(self, result):
        super(TotalAccuracy, self).__init__(result)

    def compute(self):
        self.tot_acc = self.result.sum() / self.n_total

        return self.tot_acc


class AverageForgetting(Metric):
    def __init__(self, result):
        super(AverageForgetting, self).__init__(result)

    def compute(self):
        self.avg_forget = 0
        for i in range(self.n_task):
            last = self.n_task-1
            first = i
            self.avg_forget += abs(self.result[first, i] - self.result[last, i])

        self.avg_forget = self.avg_forget / (self.n_task-1)

        return self.avg_forget


class TotalForgetting(Metric):
    def __init__(self, result):
        super(TotalForgetting, self).__init__(result)

    def compute(self):
        self.tot_forget = 0
        for i in range(self.n_task):
            for j in range(0, self.n_task-i-1):
                fore = i
                back = i+j+1
                self.tot_forget += abs(self.result[fore, i] - self.result[back, i])

        self.n_forget_total = self.n_task * (self.n_task - 1) / 2.0
        self.tot_forget = self.tot_forget / self.n_forget_total

        return self.tot_forget
