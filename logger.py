from enum import Enum
from rich.console import Console


class LogLevel(Enum):
    NONE = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    VERBOSE = 4


class Logging:
    log_level = LogLevel.NONE
    console = Console()

    @staticmethod
    def set_log_level(log_level: LogLevel):
        Logging.log_level = log_level

    @staticmethod
    def log(log_level: LogLevel, text: str):
        match Logging.log_level:
            case LogLevel.ERROR:
                match log_level:
                    case LogLevel.ERROR:
                        Logging.console.log(f"(E) : {text}", style="red")
            case LogLevel.WARNING:
                match log_level:
                    case LogLevel.ERROR:
                        Logging.console.log(f"(E) : {text}", style="red")
                    case LogLevel.WARNING:
                        Logging.console.log(f"(W) : {text}", style="yellow")
            case LogLevel.INFO:
                match log_level:
                    case LogLevel.ERROR:
                        Logging.console.log(f"(E) : {text}", style="red")
                    case LogLevel.WARNING:
                        Logging.console.log(f"(W) : {text}", style="yellow")
                    case LogLevel.INFO:
                        Logging.console.log(f"(I) : {text}")
            case LogLevel.VERBOSE:
                match log_level:
                    case LogLevel.ERROR:
                        Logging.console.log(f"(E) : {text}", style="red")
                    case LogLevel.WARNING:
                        Logging.console.log(f"(W) : {text}", style="yellow")
                    case LogLevel.INFO:
                        Logging.console.log(f"(I) : {text}")
                    case LogLevel.VERBOSE:
                        Logging.console.log(f"(V) : {text}", style="grey37")


if __name__ == "__main__":
    Logging.set_log_level(LogLevel.INFO)
    Logging.log(LogLevel.INFO, "Hello, World!")
