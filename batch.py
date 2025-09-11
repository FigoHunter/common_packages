from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread, Event
from copy import deepcopy

class BatchLoader(ABC):
    THREADS = 8
    CACHECOUNT = 32
    

    def __init__(self, file_list=[]):
        self._file_list = [*file_list]
        self._threads = []
        self._data = {}
        self._locks = {}
        self._queue = Queue(maxsize=self.CACHECOUNT)
        self._yielded_id = -1
        self._loaded_count = 0
        self._has_task = False
        for t in range(self.THREADS):
            thread = Thread(target=self._thread_loop, args=(t, self._queue))
            thread.daemon = True
            self._threads.append(thread)
            thread.start()


    def _thread_loop(self, thread_id, queue:Queue):
        while True:
            filepath = queue.get()
            if filepath is None:
                break
            data = self.load(filepath)
            self._data[filepath] = data
            self._locks[filepath].set()
            # print(f'Thread {thread_id} loaded {filepath}')
            queue.task_done()

    def _load_to_cache(self):
        for i in range(self._loaded_count, self._yielded_id + self.CACHECOUNT + 1):
            if i >= len(self._file_list):
                break
            f = self._file_list[i]
            self._locks[f] = Event()
            self._locks[f].clear()
            self._queue.put(f)
            self._has_task = True
            # print(f'Preloading {i} ...')
            self._loaded_count += 1

    def clear(self):
        while not self._queue.empty():
            self._queue.get()
            self._queue.task_done()
        if self._has_task:
            self._queue.join()
        self._data.clear()
        self._locks.clear()
        self._loaded_count = 0
        self._yielded_id = -1
        self._has_task = False

        

    def __iter__(self):
        self.clear()
        self._load_to_cache()
        for id, f in enumerate(self._file_list):
            self._locks[f].wait()
            data = self._data[f]
            self._yielded_id = max(self._yielded_id, id)
            self._load_to_cache()
            del self._data[f]
            del self._locks[f]
            # print(f'Yielding {id} ...')
            yield f, data

    def __len__(self):
        return len(self._file_list)

    @abstractmethod
    def load(self, filepath: str):
        pass

class BatchDumper(ABC):
    THREADS = 8
    CACHECOUNT = 32

    def __init__(self):
        self._threads = []
        self._queue = Queue(maxsize=self.CACHECOUNT)
        self._has_task = False
        for t in range(self.THREADS):
            thread = Thread(target=self._thread_loop, args=(t, self._queue))
            thread.daemon = True
            self._threads.append(thread)
            thread.start()

    def _thread_loop(self, thread_id, queue:Queue):
        while True:
            item = queue.get()
            if item is None:
                break
            self.dump(*item)
            queue.task_done()

    def queue_dump(self, filepath: str, data):
        data = deepcopy(data)
        self._queue.put((filepath, data))
        self._has_task = True

    def join(self):
        if self._has_task:
            self._queue.join()

    @abstractmethod
    def dump(self, filepath: str, data):
        pass