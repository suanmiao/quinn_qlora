import logging

class StringFormatter(logging.Formatter):
    def format(self, record):
        # ensure msg is a string
        record.msg = str(record.msg)
        return super().format(record)


singleton_logger = None

def get_logger():
    global singleton_logger
    if singleton_logger is None:
        singleton_logger = logging.getLogger(__name__)
        singleton_logger.setLevel(logging.INFO)
        formatter = StringFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        singleton_logger.addHandler(ch)
    return singleton_logger