"""A simple key/value store API. Like shelve but uses JSON instead of
pickle (so data must be simple structures instead of arbitrary Python
objects).
"""
import datetime
import json
import collections
import os
import sqlite3

try:
    import cPickle as pickle
except ImportError:
    import pickle


class JSONShelf(collections.MutableMapping):
    # Object lifetime.

    def save(self):
        """Persist the current in-memory state of the mapping."""
        raise NotImplementedError

    def close(self):
        """Close any opened resources."""
        pass

    def __del__(self):
        self.close()

    # As a context manager, the shelf saves on exit.

    def __enter__(self):
        # Eventually, this might acquire a lock on the file.
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.save()


class MemoryShelf(JSONShelf):
    """A base class for shelves that keep all their data in memory and
    write everything to disk on save().
    """

    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        if os.path.exists(self.filename):
            self.load()
        else:
            # Create an empty file. Good idea? Maybe not. It's how
            # SQLite works, anyway.
            self.save()

    def load(self):
        """Dump self.data to the file."""
        raise NotImplementedError

    # Pass mapping methods on to underlying store.
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()


class FlatShelf(MemoryShelf):
    """A shelf backed by a single flat JSON file.
    """

    def load(self):
        with open(self.filename) as f:
            self.data = json.load(f)

    def save(self):
        with open(self.filename, 'w') as f:

            json.dump(self.data, f, default=default)


class PickleShelf(MemoryShelf):
    """A shelf that stores data in a pickle file.
    """

    def load(self):
        with open(self.filename, 'rb') as f:
            self.data = pickle.load(f)

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.data, f)


class SQLiteShelf(JSONShelf):
    """A shelf backed by an SQLite database.
    """

    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS jsonshelve (key TEXT, value TEXT)"
        )

    def close(self):
        self.conn.close()

    def save(self):
        self.conn.commit()

    def __getitem__(self, key):
        c = self.conn.execute(
            "SELECT value FROM jsonshelve WHERE key = ? LIMIT 1", (key,)
        )
        row = c.fetchone()
        c.close()
        if not row:
            raise KeyError()
        return json.loads(row[0])

    def __setitem__(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO jsonshelve (key, value) VALUES (?, ?)",
            (key, json.dumps(value))
        )

    def __delitem__(self, key):
        c = self.conn.execute(
            "DELETE FROM jsonshelve WHERE key = ?", (key,)
        )
        if not c.rowcount:
            # No row was deleted.
            raise KeyError()

    def __iter__(self):
        c = self.conn.execute("SELECT key FROM jsonshelve")

        # This is somewhat dangerous: we allow client code to execute
        # while the cursor is still open. This means it's possible for
        # the client to create "dangling" cursors that lock the database
        # indefinitely. For this reason, this iterator should *only* be
        # used in a for loop or immediately passed to list() to avoid
        # exceptional conditions causing inadvertent locking.
        for row in c:
            yield row[0]
        c.close()

        # The less-efficient but safer alternative is:
        # return iter([r[0] for r in c])

    def __len__(self):
        c = self.conn.execute(
            "SELECT COUNT(*) FROM jsonshelve"
        )
        return c.fetchone()[0]
