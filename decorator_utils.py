from datetime import datetime


def timeit(method):
    def timed(*args, **kw):
        ts = datetime.now()
        result = method(*args, **kw)
        te = datetime.now()
        print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, (te - ts).total_seconds()))
        return result

    return timed
