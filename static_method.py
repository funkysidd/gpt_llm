from enum import Enum

class LoggingLevel(Enum):
    ERROR =   0
    WARNING = 1
    INFO =    2
    VERBOSE = 3

class Logging:
    @staticmethod
    def log(logging_level:LoggingLevel, text:str):
        match logging_level:
            case LoggingLevel.ERROR:
                print(f"Log (E) : {text}")
            case LoggingLevel.WARNING:
                print(f"Log (W) : {text}")
            case LoggingLevel.INFO:
                print(f"Log (I) : {text}")
            case LoggingLevel.VERBOSE:
                print(f"Log (V) : {text}")

if __name__ == '__main__':
    Logging.log(LoggingLevel.INFO, "Hello, World!")
