import os
import time
from functools import wraps


def delete_checkpoints(save_dir: str) -> None:
    '''Deletes all checkpoints in a directory.'''
    for file in os.listdir(save_dir):
        if file.endswith(".ckpt"):
            os.remove(os.path.join(save_dir, file))
    print(f"Deleted all checkpoints in {save_dir}")


def check_exist_dir(save_dir: str) -> None:
    '''Checks if the save directory exists.'''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory {save_dir}")


def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper
