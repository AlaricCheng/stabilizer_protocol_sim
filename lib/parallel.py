import os
import sys
import uuid
import inspect

from multiprocessing import Pool
from .tqdm_parallel import process_map, serial_map

__all__ = ['Task', 'pmap', 'tpmap', 'global_settings']


_global_settings = dict(
    processes=None
)
def global_settings(**kwargs):
    global _global_settings
    if kwargs:
        _global_settings.update(kwargs)
    return _global_settings


def is_parallel():
    p = _global_settings.get("processes", None)
    return p is None or p > 1


def Task(func):
    # return func # For aer
    
    if len(inspect.signature(func).parameters) == 0:
        def result(*args):
            return func()
    else:
        def result(*args):
            return func(*args)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def pmap(fn, iterable, **kwargs):
    settings = global_settings().copy()
    settings.update(kwargs)
    processes = settings.get("processes", None)
    chunksize = settings.get("chunksize", 1)
    initializer = settings.get('initializer', None)
    initargs = settings.get('initargs', [])

    if is_parallel():
        with Pool(processes=processes, initializer=initializer, initargs=initargs) as pool:
            return list(pool.map(fn, iterable, chunksize=chunksize))
    else:
        if initializer: initializer(*initargs)
        return list(map(fn, iterable))


def tpmap(fn, iterable, **kwargs):
    settings = global_settings().copy()
    settings.update(kwargs)

    if is_parallel():
        return process_map(fn, iterable, **settings)
    else:
        return serial_map(fn, iterable, **settings)

