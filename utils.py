import logging


def get_logger(file: str = 'log'):
    # 创建logger对象
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    # 创建FileHandler并设置日志格式、保存路径等参数
    file_handler = logging.FileHandler(file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 添加FileHandler到logger对象
    logger.addHandler(file_handler)
    return logger


