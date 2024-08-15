class Config:
    n_jobs = 5  # Default value
    batch_size = 1000  # Default value    

config = Config()

def set_n_jobs(n):
    config.n_jobs = n

def set_batch_size(n):
    config.batch_size = n