import logging
from logging.handlers import RotatingFileHandler
import os

class AppLogger:
    def __init__(self, name, log_file='app.log', max_bytes=2000, backup_count=0, level=logging.DEBUG):
        logger = logging.getLogger(name)
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

        self.logger = logger
        self.logger.setLevel(level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        if os.path.exists(log_file):
            os.remove(log_file)
            
        fh = RotatingFileHandler(log_file, mode='a', maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)