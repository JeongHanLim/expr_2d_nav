import os

class ExperimentManager(object):
    def __init__(self, path, mid_class, sub_class):
        self.path = path
        self.mid_class = mid_class
        self.sub_class = sub_class
        self.mid_path = os.path.join(path, mid_class)
        self.sub_path = os.path.join(os.path.join(path, mid_class), sub_class)
        if not os.path.isdir(self.mid_path):
            os.mkdir(self.mid_path)
        if not os.path.isdir(self.sub_path):
            os.mkdir(self.sub_path)

    def make_description(self, description):
        if not os.path.isfile(os.path.join(self.mid_path, 'description.csv')):
            with open(os.path.join(self.mid_path, 'description.csv'), 'w') as f:
                f.write('subclass, description\n')
                f.write('{}, {}\n'.format(self.sub_class, description))
        else:
            with open(os.path.join(self.mid_path, 'description.csv'), 'a') as f:
                f.write('{}, {}\n'.format(self.sub_class, description))
        return