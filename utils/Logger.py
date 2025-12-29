import logging
import os


def get_logger(file_name, data_name, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    filename = f"{log_path}/" + data_name + ".log"
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
