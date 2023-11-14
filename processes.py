import signal
import select
import sys
import os
import stat
import threading
import time
from datetime import datetime

from tensorflow.keras.callbacks import Callback

import gravyflow as gf

def explain_exit_code(exit_code):
    """
    Return a string explaining the meaning of a given exit code.

    Args:
    exit_code (int): The exit code to explain.

    Returns:
    str: A string explanation of the exit code.
    """
    if exit_code == 0:
        return "Success"
    elif exit_code < 0:
        signal_num = abs(exit_code)
        signal_name = signal.Signals(signal_num).name
        return f"Terminated by signal {signal_num} ({signal_name})"
    else:
        # Common Unix Exit Codes
        common_exit_codes = {
            1: "General error, unspecified",
            2: "Misuse of shell builtins (according to Bash documentation)",
            126: "Command invoked cannot execute",
            127: "Command not found",
            128: "Invalid exit argument.",
            130: "Script terminated by Control-C (SIGINT)",
            137: "Process killed (SIGKILL or similar)",
            139: "Segmentation fault",
            143: "Terminated by signal 15 (SIGTERM)",
        }
        return common_exit_codes.get(exit_code, "Unknown error")

def signal_handler(signum, frame):
    """
    Signal handler function.
    """
    sys.stderr.write(f"Received termination signal: {signum} : {explain_exit_code(signum)}. Exiting.\n")
    # Perform any clean-up tasks here
    sys.exit(signum)

# Function to clean up the named pipe
def cleanup_named_pipe(pipe_name):
    try:
        os.remove(pipe_name)
    except OSError:
        pass

def check_if_pipe_exists(pipe_path):
    # Check if the path exists
    if os.path.exists(pipe_path):
        # Check if the path is a named pipe (FIFO)
        mode = os.stat(pipe_path).st_mode
        if stat.S_ISFIFO(mode):
            return True
        else:
            return False
    else:
        return False

def create_named_pipe(pipe_name):
    # Check if the named pipe already exists

    error = "Unknown" 

    gf.ensure_directory_exists("tmp")

    if os.path.exists(pipe_name):
        # Remove the existing pipe before creating a new one
        os.remove(pipe_name)
        os.mkfifo(pipe_name)
    else:
        try:
            os.mkfifo(pipe_name)
            print(f"Named pipe {pipe_name} created.")
        except OSError as e:
            print(f"Failed to create named pipe: {e}")

            error = e


    if not check_if_pipe_exists(pipe_name):
        print(f"Failed to create named pipe: {e}")


def write_non_blocking(pipe_name, message):
    try:
        # Open the named pipe in non-blocking mode for writing
        fd = os.open(pipe_name, os.O_WRONLY | os.O_NONBLOCK)
        with os.fdopen(fd, 'w') as fifo_writer:
            try:
                fifo_writer.write(message)
                print("Message written successfully.")
            except OSError as e:
                if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                    print("Write operation would block, no reader available.")
                else:
                    raise  # Re-raise the exception if it's not a 'would block' error
    except FileNotFoundError:
        print(f"Named pipe {pipe_name} does not exist.")
    except Exception as e:
        print(f"Error opening/writing to pipe: {e}")

def parse_name_and_time(input_string):
    # Split the input string into name and timestamp
    parts = input_string.split(':')
    if len(parts) != 2:
        raise ValueError("Input string must be in the format 'name:timestamp'")
    
    name, timestamp_str = parts

    # Convert the timestamp string to a datetime object
    try:
        timestamp = float(timestamp_str)
    except ValueError:
        raise ValueError("Timestamp is not a valid float")

    return name, timestamp

def check_heartbeat_integrity(heartbeat, expected_command_name):
    """
    Checks the integrity of a heartbeat message.

    :param heartbeat: The heartbeat message string.
    :param expected_command_name: The expected name in the heartbeat message.
    :return: A tuple (is_valid, timestamp). is_valid is a boolean indicating
             whether the heartbeat is valid. timestamp is the parsed timestamp
             if valid, otherwise None.
    """
    name, timestamp = parse_name_and_time(heartbeat)

    if not name or not timestamp:
        print("Malformed heartbeat, assumed dead.")
        if not name:
            print("Heartbeat name is missing!")
        if not timestamp:
            print("Could not convert timestamp to float!")
        return None, None

    if name != expected_command_name:
        print("Malformed heartbeat, assumed dead.")
        print("Heartbeat name does not match!")
        return None, None

    return name, timestamp

def open_non_blocking(pipe_name):

    # Open the named pipe in non-blocking mode
    try:
        fd = os.open(pipe_name, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError:
        print(f"Named pipe {pipe_name} does not exist. Subprocess might have terminated.")
        return None
    
    return open(fd, 'r')

def acquire_heartbeat(
        command,
        acquisition_timeout_seconds
    ):

    try:
        with open_non_blocking(command.pipe_name) as fifo_reader:
            ready, _, _ = select.select([fifo_reader], [], [], acquisition_timeout_seconds) 

            if ready:
                heartbeat = fifo_reader.read()
                if heartbeat:
                    
                    name, timestamp = check_heartbeat_integrity(
                        heartbeat, 
                        command.name
                    )

                    if (name is not None) and (timestamp is not None):
                        return timestamp
                    else:
                        return None
                else:
                    print(f"No heartbeat received from {command.name}.")
                    return None
            else:
                return 0

    except FileNotFoundError:
        print(f"Named pipe {command.pipe_name} does not exist. Subprocess might have terminated.")
        return None
    except Exception as e:
        print(f"Error reading from pipe for {command.pipe_name}: {e}")
        return None
 
def monitor_heartbeat(
        command, 
        flags,
        missed_heartbeat_threshold=1200,
        acquisition_timeout_seconds=60
    ):

    """
    Monitor the heartbeat of a subprocess.

    :param command: The command object representing the subprocess.
    :param missed_heartbeat_threshold: Number of missed heartbeats to trigger an action.
    """
    if not flags["should_exit"].is_set():
        
        print(f"Acquiring heartbeat {command.name} at {command.id}...")
        last_heartbeat_timestamp = acquire_heartbeat(
            command,
            acquisition_timeout_seconds=acquisition_timeout_seconds
        )
        if last_heartbeat_timestamp is None or last_heartbeat_timestamp == 0:
            print("Failed to acquire heartbeat! Assumed dead!")
            flags["has_died"].set()
            return -1
        else:
            print(f"{command.name} at {command.id} heartbeat acquired at: {last_heartbeat_timestamp} s.")
    else:
        return -1

    while not flags["should_exit"].is_set():
        timestamp = acquire_heartbeat(
            command,
            acquisition_timeout_seconds=5
        )

        if timestamp != 0:
            last_heartbeat_timestamp = timestamp
        elif timestamp is None:
            flags["has_died"].set()
            return -1
        elif timestamp == -1:
            return 0

        time_since_last_beat = time.time() - last_heartbeat_timestamp
        if time_since_last_beat >= missed_heartbeat_threshold:
            print(f"It has been {time_since_last_beat} seconds since last heartbeat detected from {command.name}.")
            flags["has_died"].set()
            return -1

        if flags["should_exit"].is_set():
            return -1

        time.sleep(1)
    
    return -1
        
def start_monitoring_thread(command, flags):
    """
    Start a new thread to monitor the heartbeat of a subprocess.

    :param command: The command object representing the subprocess.
    """
    monitor_thread = threading.Thread(target=monitor_heartbeat, args=(command, flags))
    monitor_thread.start()

    return monitor_thread

def kill_process(pid):
    try:
        os.kill(pid, signal.SIGKILL)  # or signal.SIGKILL for a forceful kill
        print(f"Process with PID {pid} has been terminated.")
    except OSError as e:
        return

class Heart:
    def __init__(self, pipe_name : str):
        self.pipe_name = pipe_name
        self.beat()

    def beat(self):
        print("Boom boom.")
        write_non_blocking(
            f"./tmp/heartbeat_{self.pipe_name}", f"{self.pipe_name}:{str(time.time())}"
        )

# Keras callback
class HeartbeatCallback(Callback):
    def __init__(self, heart, interval):
        super().__init__()
        self.heart = heart
        self.interval = interval

    def on_batch_end(self, batch, logs=None):

        print("Here!")
        if batch % self.interval == 0:
            self.heart.beat()