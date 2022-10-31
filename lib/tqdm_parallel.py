"""
Thin wrappers around `multiprocessing`.
"""
from __future__ import absolute_import

from contextlib import contextmanager

from tqdm.auto import tqdm as tqdm_auto

try:
    from operator import length_hint
except ImportError:
    def length_hint(it, default=0):
        """Returns `len(it)`, falling back to `default`"""
        try:
            return len(it)
        except TypeError:
            return default

__author__ = {"github.com/": ["casperdcl", "xlthu"]}
__all__ = ['thread_map', 'process_map']


@contextmanager
def ensure_lock(tqdm_class, lock_name=""):
    """get (create if necessary) and then restore `tqdm_class`'s lock"""
    old_lock = getattr(tqdm_class, '_lock', None)  # don't create a new lock
    lock = old_lock or tqdm_class.get_lock()  # maybe create a new lock
    lock = getattr(lock, lock_name, lock)  # maybe subtype
    tqdm_class.set_lock(lock)
    yield lock
    if old_lock is None:
        del tqdm_class._lock
    else:
        tqdm_class.set_lock(old_lock)


def _initializer(tqdm_class, lk, initializer, *initargs):
    tqdm_class.set_lock(lk)
    if initializer: initializer(*initargs)


def _executor_map(pool, fn, iterable, *, tqdm_class, processes, chunksize, total, lock_name, **kwargs):
    """
    Implementation of `thread_map` and `process_map`.
    """
    kwargs["total"] = total
    if total is None:
        kwargs["total"] = length_hint(iterable)

    initializer = kwargs.pop('initializer', None)
    initargs = kwargs.pop('initargs', [])

    with ensure_lock(tqdm_class, lock_name=lock_name) as lk:
        with pool(
            processes=processes,
            initializer=_initializer,
            initargs=[tqdm_class, lk, initializer] + initargs
        ) as ex:
            return list(tqdm_class(ex.imap(fn, iterable, chunksize=chunksize), **kwargs))


def thread_map(fn, iterable, *, tqdm_class=tqdm_auto, processes=None, chunksize=1, total=None, lock_name='mp_lock', **kwargs):
    """
    See process_map
    But use thread
    """
    from multiprocessing.pool import ThreadPool
    return _executor_map(ThreadPool, fn, iterable, tqdm_class=tqdm_class, processes=processes, chunksize=chunksize, total=total, lock_name=lock_name, **kwargs)


def process_map(fn, iterable, *, tqdm_class=tqdm_auto, processes=None, chunksize=1, total=None, lock_name='mp_lock', **kwargs):
    """
    Equivalent of `list(map(fn, iterable))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    processes  : int, optional
        The number of worker processes to use. If processes is 
        None then the number returned by os.cpu_count() is used.
    chunksize  : int, optional
        Size of chunks sent to worker processes; passed to
        `multiprocessing.Pool.map`. [default: 1].
    total : int, optional
        Hint on the length of iterable
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    **kwargs : optional
        Passed to `tqdm_class`
    """
    from multiprocessing import Pool
    return _executor_map(Pool, fn, iterable, tqdm_class=tqdm_class, processes=processes, chunksize=chunksize, total=total, lock_name=lock_name, **kwargs)


def serial_map(fn, iterable, *, tqdm_class=tqdm_auto, total=None, **kwargs):
    kwargs.pop('processes', None)
    kwargs.pop('chunksize', None)
    kwargs.pop('lock_name', None)

    kwargs["total"] = total
    if total is None:
        kwargs["total"] = length_hint(iterable)

    initializer = kwargs.pop('initializer', None)
    initargs = kwargs.pop('initargs', [])
    if initializer: initializer(*initargs)
    
    return list(map(fn, tqdm_class(iterable, **kwargs)))
