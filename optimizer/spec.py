class OptimizerSpec(object):
    def __init__(self, optimizer, d_in):
        self.optimizer = optimizer
        self.d_in = d_in


class LearningSpec(object):
    def __init__(self, n_epoch, n_batch, n_train, n_task, model_dir, optimizer_spec, n_fed_step, n_fed_round, alpha=1.0):
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.alpha = alpha
        self.n_task = n_task
        self.optimizer_spec = optimizer_spec
        self.model_dir = model_dir
        self.alpha = alpha
        self.n_train = n_train
        self.n_fed_step = n_fed_step
        self.n_fed_round = n_fed_round
