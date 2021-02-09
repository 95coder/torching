import os
import logging
from logging import handlers
from functools import partial


FMT_TNLFM = '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
FMT_TNLM = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
FMT_TLM = '%(asctime)s | %(levelname)s | %(message)s'
FMT_TM = '%(asctime)s | %(message)s'
FMT_M = '%(message)s'


class Singleton(type):
    def __init__(self, *args, **kwargs):
        self._instance = None
        super(Singleton, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = super(Singleton, self).__call__(*args, **kwargs)
            return self._instance
        else:
            return self._instance


class create_logger(metaclass=Singleton):
    __loggers = {}

    @classmethod
    def __call__(cls,
                 name='root',
                 level=logging.INFO,
                 stream_handler=None,
                 file_handler=None,
                 **options):
        if not name:
            return None

        if name in cls.__loggers:
            return cls.__loggers.get(name)

        if name == 'root':
            logger = logging.root
        else:
            logger = logging.getLogger(name)
            
        logger.setLevel(level=level)
        logger.propagate = False

        if stream_handler is not None:
            logger.addHandler(stream_handler)
        else:
            stream_handler = cls.create_stream_handler(level, FMT_TNLM)
            logger.addHandler(stream_handler)

        if file_handler is not None:
            logger.addHandler(file_handler)

        cls.__loggers[name] = logger

        return logger

    @staticmethod
    def create_stream_handler(level, fmt):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter(fmt))
        return stream_handler

    @staticmethod
    def create_file_handler(level, fmt, filename, maxbytes=100*1024**2, backupcount=5):
        logdir, _ = os.path.split(filename)

        if not os.path.isdir(logdir):
            os.makedirs(logdir)

        file_handler = handlers.RotatingFileHandler(filename, maxBytes=maxbytes, backupCount=backupcount)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))

        return file_handler