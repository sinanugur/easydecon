from dataclasses import dataclass


@dataclass
class Config:
    n_jobs: int = 5
    batch_size: int | str = 1000

config = Config()

def set_n_jobs(n):
    n = int(n)
    if n == 0:
        raise ValueError("n_jobs cannot be 0. Use 1 for serial or -1 for all CPUs.")
    config.n_jobs = n

def set_batch_size(n):
    if n == "auto":
        config.batch_size = n
        return
    n = int(n)
    if n < 1:
        raise ValueError("batch_size must be a positive integer or 'auto'.")
    config.batch_size = n
