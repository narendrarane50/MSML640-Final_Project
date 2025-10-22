import os, sys, datetime
from pathlib import Path
from typing import Optional

class Logger:
    def __init__(self, log_dir: str = "logs", filename: Optional[str] = None):
        Path(log_dir).mkdir(exist_ok=True)
        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"train_{ts}.log"
        self.log_path = os.path.join(log_dir, filename)
        self.log_file = open(self.log_path, "a")
        self.start_time = datetime.datetime.now()
        self.write(f"[Logger] Started at {self.start_time}\n")

    def write(self, msg: str, also_stdout: bool = True):
        if also_stdout:
            print(msg.strip())
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        end = datetime.datetime.now()
        self.write(f"[Logger] Finished at {end} (Duration: {end - self.start_time})")
        self.log_file.close()