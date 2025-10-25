from enum import Enum


class LogLevel(Enum):
    NONE = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    VERBOSE = 4


class Logging:
    log_level = LogLevel.NONE

    @staticmethod
    def set_log_level(log_level: LogLevel):
        Logging.log_level = log_level

    @staticmethod
    def log(log_level: LogLevel, text: str):
        match Logging.log_level:
            case LogLevel.ERROR:
                match log_level:
                    case LogLevel.ERROR:
                        print(f"Log (E) : {text}")
            case LogLevel.WARNING:
                match log_level:
                    case LogLevel.ERROR:
                        print(f"Log (E) : {text}")
                    case LogLevel.WARNING:
                        print(f"Log (W) : {text}")
            case LogLevel.INFO:
                match log_level:
                    case LogLevel.ERROR:
                        print(f"Log (E) : {text}")
                    case LogLevel.WARNING:
                        print(f"Log (W) : {text}")
                    case LogLevel.INFO:
                        print(f"Log (I) : {text}")
            case LogLevel.VERBOSE:
                match log_level:
                    case LogLevel.ERROR:
                        print(f"Log (E) : {text}")
                    case LogLevel.WARNING:
                        print(f"Log (W) : {text}")
                    case LogLevel.INFO:
                        print(f"Log (I) : {text}")
                    case LogLevel.VERBOSE:
                        print(f"Log (V) : {text}")


if __name__ == "__main__":
    Logging.set_log_level(LogLevel.INFO)
    Logging.log(LogLevel.INFO, "Hello, World!")
