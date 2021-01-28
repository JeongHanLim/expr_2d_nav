class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs):
        self.lr = base_lr
        self.epoch = num_epochs

    def __call__(self, optimizer, epoch):
        # lr = self.lr * (1 - (epoch / self.epoch / 1.33))
        lr = self.lr
        # warm up lr schedule
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0
        optimizer.param_groups[0]['lr'] = lr