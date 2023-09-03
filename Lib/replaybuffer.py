import numpy as np

class ReplayBuffer(object):
    """回放缓冲区"""
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
    
    def put(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get(self, batch_size):
        sample_size = min(batch_size, self.size)
        samples = np.random.choice(len(self.buffer), size=sample_size, replace=False)
        experiences = [self.buffer[i] for i in samples]

        return experiences
    
    @property
    def size(self):
        return len(self.buffer)
    
