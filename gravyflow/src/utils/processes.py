import signal
import select
import sys
import os
import stat
import threading
import subprocess
import copy
import logging
import time
from typing import List
from datetime import datetime
from pathlib import Path

from tensorflow.keras.callbacks import Callback
import numpy as np
import pytz

import gravyflow as gf

def get_current_datetime():
    """Returns the current datetime in UK timezone as a formatted string."""
    uk_timezone = pytz.timezone('Europe/London')
    uk_time = datetime.now(uk_timezone)
    return uk_time.strftime("%Y-%m-%d %H:%M:%S")

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
    if pipe_name is not None:
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
            logging.info(f"Named pipe {pipe_name} created.")
        except OSError as e:
            logging.error(f"Failed to create named pipe: {e}")

            error = e

    if not check_if_pipe_exists(pipe_name):
        logging.error(f"Failed to create named pipe: {e}")


def write_non_blocking(pipe_name, message):
    try:
        # Open the named pipe in non-blocking mode for writing
        fd = os.open(pipe_name, os.O_WRONLY | os.O_NONBLOCK)
        with os.fdopen(fd, 'w') as fifo_writer:
            try:
                fifo_writer.write(message)
            except OSError as e:
                if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                    logging.error("Write operation would block, no reader available.")
                else:
                    raise  # Re-raise the exception if it's not a 'would block' error
    except FileNotFoundError:
        logging.error(f"Named pipe {pipe_name} does not exist.")
    except Exception as e:
        logging.error(f"Error opening/writing to pipe: {e}")

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
        logging.error("Malformed heartbeat, assumed dead.")
        if not name:
            logging.error("Heartbeat name is missing!")
        if not timestamp:
            logging.error("Could not convert timestamp to float!")
        return None, None

    if name != expected_command_name:
        logging.error("Malformed heartbeat, assumed dead.")
        logging.error("Heartbeat name does not match!")
        return None, None

    return name, timestamp

def open_non_blocking(pipe_name):

    # Open the named pipe in non-blocking mode
    try:
        fd = os.open(pipe_name, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError:
        logging.error(f"Named pipe {pipe_name} does not exist. Subprocess might have terminated.")
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
                        logging.warning(f"Malformed heartbeat received from {command.name}.")
                        return 0
                else:
                    return 0
            else:
                return 0

    except FileNotFoundError:
        logging.error(f"Named pipe {command.pipe_name} does not exist. Subprocess might have terminated.")
        return None
    except Exception as e:
        logging.error(f"Error reading from pipe for {command.pipe_name}: {e}")
        return 0
 
def monitor_heartbeat(
        command, 
        flags,
        missed_heartbeat_threshold=300,
        acquisition_timeout_seconds=60
    ):

    """
    Monitor the heartbeat of a subprocess.

    :param command: The command object representing the subprocess.
    :param missed_heartbeat_threshold: Number of missed heartbeats to trigger an action.
    """
    if not flags["should_exit"].is_set():
        
        logging.info(f"Acquiring heartbeat {command.name}...")
        last_heartbeat_timestamp = acquire_heartbeat(
            command,
            acquisition_timeout_seconds=acquisition_timeout_seconds
        )
        if last_heartbeat_timestamp is None or last_heartbeat_timestamp == 0:
            logging.warning(f"Failed to acquire heartbeat! Assumed dead at {get_current_datetime()}!")
            flags["has_died"].set()
            return -1
        else:
            logging.info(f"{command.name} at {command.id} heartbeat acquired at {get_current_datetime()}.")
            flags["heartbeat_acquired"].set()
    else:
        return -1

    while not flags["should_exit"].is_set():
        timestamp = acquire_heartbeat(
            command,
            acquisition_timeout_seconds=missed_heartbeat_threshold
        )

        if flags["should_exit"].is_set():
            return -1
        
        if timestamp is None:
            flags["has_died"].set()
            return -1

        if timestamp == -1:
            logging.info(f"{command.name} has succesfully completed at {get_current_datetime()}. Enforcing shutdown.")
            time.sleep(30)
            flags["has_completed"].set()
            return 0
        elif timestamp != 0:
            last_heartbeat_timestamp = timestamp

        try:
            time_since_last_beat = time.time() - last_heartbeat_timestamp
        except:
            logging.warning("Malformed timestamp detected.")
            continue
        
        if time_since_last_beat >= missed_heartbeat_threshold:
            logging.warning(
                (f"{get_current_datetime()}: It has been {time_since_last_beat} "
                f"seconds since last heartbeat detected from {command.name}.")
            )
            flags["has_died"].set()
            return -1

        time.sleep(0.1)
        
    return -1
        
def start_monitoring_thread(command, flags):
    """
    Start a new thread to monitor the heartbeat of a subprocess.

    :param command: The command object representing the subprocess.
    """
    monitor_thread = threading.Thread(
        target=monitor_heartbeat, 
        args=(command, flags)
    )
    monitor_thread.start()

    return monitor_thread

def kill_process(pid):
    try:
        if pid is not None and pid > 0:
            os.kill(pid, signal.SIGKILL)
            logging.info(f"Process with PID {pid} has been terminated.")
    except OSError as e:
        return

class Heart:
    def __init__(self, pipe_name : str):
        self.pipe_name = pipe_name
        self.beat()

    def beat(self):
        write_non_blocking(
            f"./tmp/heartbeat_{self.pipe_name}", f"{self.pipe_name}:{str(time.time())}"
        )

    def complete(self):
        write_non_blocking(
            f"./tmp/heartbeat_{self.pipe_name}", f"{self.pipe_name}:-1"
        )

# Keras callback
class HeartbeatCallback(Callback):
    def __init__(self, heart, interval = 32):
        super().__init__()
        self.heart = heart
        self.interval = interval

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            self.heart.beat()

    def on_predict_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            self.heart.beat()
    
    def on_test_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            self.heart.beat()

class Process:
    def __init__(
            self, 
            command_string : str, 
            name : str,
            tensorflow_memory_mb : float, 
            cuda_overhead_mb : float,
            initial_restart_count : int = 1
        ):

        self.current_gpu = -1
        self.memory_assigned = 0
        self.tensorflow_memory_mb = tensorflow_memory_mb
        self.cuda_overhead_mb = cuda_overhead_mb

        self.flags = None
        self.pipe_name = None
        self.pipe_monitor_thread = None

        self.process = None
        self.id = -1
        self.last_retcode = None
        self.restart_count = initial_restart_count
        self.total_restart_count = initial_restart_count
        self.restart_counter = time.time()
        self.full = command_string
        self.name = name
        self.has_failed = False
        self.has_completed = False

        self.manager = None

        parts = command_string.split()
    
        # Check if the command starts with "python":
        if parts and parts[0] == "python":
            parts.pop(0)  # Remove 'python' from the command
        else:
            raise ValueError("Command does not start with 'python'.")

        # Extract the script path and name
        self.path = parts.pop(0)

        # Parse arguments
        self.args = {}
        current_key = None
        for part in parts:
            if part.startswith("--"):
                current_key = part[2:]
                self.args[current_key] = []
            elif current_key is not None:
                self.args[current_key].append(part)

        # Join the argument values if they were split due to spaces:
        for key, value in self.args.items():
            self.args[key] = " ".join(value)

    # Modify the start_process function to track first-time start:
    def start(self):
        try:
            process_gap_seconds = 10
            logging.info((
                f"Waiting {process_gap_seconds} s"
                " to space out process activations."
            ))
            # Space out so not too many are run at the same time:
            time.sleep(process_gap_seconds)  

            # Determine the mode for log files based on 
            # whether it's a first-time start:
            mode = "w" if self.total_restart_count == 0 else "a"
            
            log_directory_path = self.manager.log_directory_path
            gf.ensure_directory_exists(log_directory_path)

            with open(
                    f"{log_directory_path}/{self.name}_log.txt", 
                    mode
                ) as stdout_file, \
                open(
                    f"{log_directory_path}/{self.name}_error.txt",
                    mode
                ) as stderr_file:

                full_command = (
                    f"{self.full} --gpu {self.current_gpu}"
                    f" --request_memory {self.tensorflow_memory_mb}"
                    f" --restart_count {self.total_restart_count}"
                    f" --name {self.name}"
                )

                self.pipe_name = f"./tmp/heartbeat_{self.name}"
                create_named_pipe(self.pipe_name)

                self.flags = {
                    "heartbeat_acquired" : threading.Event(),
                    "has_died" : threading.Event(), 
                    "should_exit" : threading.Event(), 
                    "has_completed": threading.Event()
                }
                self.pipe_monitor_thread = start_monitoring_thread(
                    self, self.flags
                )
                
                self.process = subprocess.Popen(
                    full_command, 
                    shell=True, 
                    stdout=stdout_file, 
                    stderr=stderr_file, 
                )

                self.id = self.process.pid
                logging.info(
                    f"Process: {self.name} started at {self.id}"
                )
                
                # Start restart counter:
                self.restart_counter = time.time()

        except Exception as e:
            logging.exception((
                f"Failed to start process {self.name}"
                " on GPU {self.current_gpu}."
            ))
            self.restart_count += 1
            self.total_restart_count += 1
            return None

    def kill(self):
        
        if (self.id is not None) or (self.id != -1):
            kill_process(self.id)
        
        if self in self.manager.running:
            self.manager.running.remove(self)

        if self.manager.allocated_memory is not None:
            self.manager.allocated_memory[
                self.current_gpu
            ] -= self.memory_assigned
        else:
            logging.warning(
                "Allocated memory is None when removing process."
            )

        if self.manager.allocated_memory[
                self.current_gpu
            ] < 0:

            self.manager.allocated_memory[
                self.current_gpu
            ] = 0
            logging.warning(
                "Allocated memory has fallen below zero."
            )
        
        self.process = None
        self.id = -1
        self.current_gpu = -1
        self.memory_assigned = 0
        
        if self.flags is not None:
            self.flags["should_exit"].set()
        
        cleanup_named_pipe(self.pipe_name)

    def check_if_completed(self):

        if self.has_completed:
            self.kill()
            self.has_completed = True
            self.manager.completed.append(self)

            if self in self.manager.running:
                self.manager.running.remove(self)

            return True

        return False
    
    def check_if_failed(self):

        # Check if already failed:
        if self.has_failed or (self in self.manager.failed):
            self.kill()
            self.has_failed = True  # Mark process as failed
            self.manager.failed.append(self)

            return True
        
        # Update fail restart timer:
        current_time = time.time()
        if current_time - self.restart_counter > self.manager.restart_timeout_seconds:
            self.restart_counter = current_time
            self.restart_count = 0

        # Check if the process has been restarted more than N times in X seconds:
        if self.restart_count > self.manager.max_restarts:
            logging.error((
                f"Process {self.name} has been restarted "
                f"{self.restart_count} times within "
                f"{self.manager.restart_timeout_seconds} "
                f"seconds. Marking as failed."
            ))
            self.has_failed = True  # <-- Mark process as failed
            self.kill()
            self.manager.failed.append(self)
            return True
        
        return False      

    def get_retcode(self):
        retcode = self.process.poll()
        self.last_retcode = retcode
        return retcode

    def print_stderr(self):
        stdout, stderr = self.process.communicate()
        if stdout:
            logging.error((
                f"Process {self.name} at {self.id}"
                f" - STDOUT: {stdout.decode()}"
            ))
        if stderr:
            logging.error((
                f"Process {self.name} at {self.id}"
                f" - STDERR: {stderr.decode()}"
            ))

    def complete(self):
        self.kill()
        self.has_completed = True
        self.manager.completed.append(self)

    def requeue(
            self,
            memory_increase_mb : int = 1000, 
            max_memory_mb : int = 8000
        ):

        self.restart_count += 1
        self.total_restart_count += 1
        if not self.check_if_failed():
            self.manager.queued.insert(0, self)
            self.manager.num_restarts += 1
            self.kill()

        match self.last_retcode:
            case -6:
                if self.manager.max_num_concurent_processes > 20:
                    logging.info(
                        (
                            f"Process {self.name} was killed. Reducing max number of processes from "
                            f"{self.manager.max_num_concurent_processes} to {self.manager.max_num_concurent_processes - 1}."
                        )
                    )
                    self.manager.max_num_concurent_processes -= 1
            case 1:
                logging.info(
                    (
                        f"Process {self.name} ended with general error, assuming OOM error. Increasing job memory requriment from "
                         f"{self.tensorflow_memory_mb} to {self.tensorflow_memory_mb + memory_increase_mb}."
                    )
                )
            case _:
                pass
        
        self.tensorflow_memory_mb += memory_increase_mb
        self.cuda_overhead_mb += memory_increase_mb // 2

        if self.tensorflow_memory_mb > max_memory_mb:
            logging.info(
                (
                    f"Process {self.name} has reached max memory requirement {max_memory_mb} "
                    f"it will fail if this is not enough."
                )
            )
            self.tensorflow_memory_mb = max_memory_mb
        
        self.memory_assigned = self.tensorflow_memory_mb + self.cuda_overhead_mb

class Manager:

    queued = []
    running = []
    failed = []
    completed = []
    all = []
    
    free_memory = None
    allocated_memory = None

    num_restarts = 0
    num_iterations = 0

    num_queued = 0
    num_running = 0
    num_failed = 0
    num_completed = 0
    total = 0

    def tabulate(self):

        # Calculate the number of lines to go back up in the terminal
        lines_to_move_up = len(self.all) + 2  # +2 for headers and an extra line

        # Clear the lines
        for _ in range(lines_to_move_up):
            print("\033[F\033[K", end="")
        
        # Print table headers
        header = (
            f"| {'ID':<7} | {'Name':<15} | {'GPU':<6} | {'Restart Timeout': <15} "
            f"| {'Total Restarts': <15} | {'Assigned Mem':<15} | {'TF Mem':<10} "
            f"| {'CUDA Overhead':<15} | {'Status':<8} |"
        )

        print(self)
        print(header)

        # Print each process in the table
        for process in self.all:
            restarts_string = f"{process.restart_count} / {self.max_restarts}"
            status = "Failed" if process.has_failed else "Completed" if process.has_completed else "Running" if process.id > 0 else "Waiting"
            row = f"| {process.id:<7} | {process.name:<15} | {process.current_gpu:<6} | {restarts_string:<15} | {process.total_restart_count:<15} | {process.memory_assigned:<15} | {process.tensorflow_memory_mb:<10} | {process.cuda_overhead_mb:<15} | {status:<8} |"
            print(row)

    def __str__(self):
        num_queued = len(self.queued)
        num_running = len(self.running)
        num_failed = len(self.failed)
        num_completed = len(self.completed)
        total = num_queued + num_running + num_failed + num_completed

        return (
            f"{num_running}/{total} running, "
            f"{num_completed}/{total} completed, "
            f"{num_failed}/{total} failed, "
            f"{num_queued}/{total} in queue. "
            f"{self.num_restarts} restarts."
        )

    def __init__(
            self, 
            initial_processes : List[Process],
            max_restarts : int = 10,
            restart_timeout_seconds : float = 1200, 
            process_start_wait_seconds : float = 1, 
            management_tick_length_seconds : float = 1,
            max_num_concurent_processes : int = 10,
            log_directory_path : Path = Path("./logs"),
            force_retrain = False
        ):

        if not isinstance(initial_processes, list):
            initial_processes = [initial_processes]

        self.queued = initial_processes
        self.all = copy.copy(initial_processes)

        for process in self.queued:
            process.manager = self

        self.force_retrain = force_retrain
        self.log_directory_path = log_directory_path
        self.max_restarts = max_restarts
        self.restart_timeout_seconds = restart_timeout_seconds
        self.max_num_concurent_processes = max_num_concurent_processes
        self.process_start_wait_seconds = process_start_wait_seconds
        self.management_tick_length_seconds = management_tick_length_seconds
        self.total_processes = 0

    def __bool__(self):
        
        if (self.queued or self.running):
            return True
        else:
            logging.info((
                f"All processes finished. "
                f"{len(self.completed)}/{self.total_processes}"
                f" completed, {len(self.failed)}/{self.total_processes} failed."
                f" {self.num_restarts} attempted restarts."
            ))
            return False

    def __call__(self):

        self.total_processes = len(self.queued) + len(self.running) \
            + len(self.completed) + len(self.failed)

        if not self.num_iterations:

            try:
                num_gpus = len(gf.get_memory_array())
            except:
                raise Exception("Cannot get num GPUs!")
            
            self.allocated_memory : np.ndarray = np.zeros([num_gpus], dtype = np.int64)

            logging.info(
                "Starting the process management system..."
            )
            logging.info((
                f"Monitoring {self.total_processes} "
                f"processes across {num_gpus} available GPUs."
            ))

        # Perform a single iteration of process management:
        self.manage_running_processes()
        self.start_processes()

        if gf.is_redirected():
            time.sleep(
                self.management_tick_length_seconds
            )

        self.num_iterations += 1

    def manage_running_processes(self):

        for process in self.running[:]:
            
            if process.process is not None:
                
                if process.check_if_failed():
                    logging.warning((
                        "Failed process found in running jobs."
                    ))
                    
                    if process in self.queued:
                        self.queued.remove(process)
                    continue
                if process.check_if_completed():
                    logging.warning((
                        "Completed process found in running jobs."
                    ))

                    if process in self.queued:
                        self.queued.remove(process)
                    continue
                
                # Manage process exit:
                retcode = process.get_retcode()
                if retcode is not None:  # Process finished.

                    process.print_stderr()
                    process.kill()

                    # Check if the process should be marked as failed
                    if retcode != 0:  # Process failed, log the error
                        logging.error((
                            f"Process {process.name} at {process.id} "
                            f"failed with return code {retcode} : "
                            f"{explain_exit_code(retcode)}. "
                             "Attempting requeue."
                        ))
                        process.requeue()
                    else:
                        logging.info((
                            f"Process {process.name} at {process.id} "
                            f"completed sucessfully with return code {retcode}: "
                            f"{explain_exit_code(retcode)}."
                        ))
                        process.complete()

                # Check if monitor thread has marked process as dead:
                elif process.flags["heartbeat_acquired"].is_set():
                    logging.info((
                        f"Monitoring thread for {process.id} communication check sucessfull."
                    ))
                    process.flags["heartbeat_acquired"].clear()

                # Check if monitor thread has spotted completion signal:
                elif process.flags["has_completed"].is_set():
                    logging.error((
                        f"Process {process.name} at {process.id}"
                        " failed to complete gracefully. Forcing termination."
                    ))
                    process.complete()

                # Check if monitor thread has marked process as dead:
                elif process.flags["has_died"].is_set():
                    logging.error((
                        f"Process {process.name} at "
                        f"{process.id} heartbeat lost. "
                        "Terminating."
                    ))
                    process.last_retcode = 1
                    process.requeue()

    def update_memory_array(self):
        try:
            self.free_memory = gf.get_memory_array()

            if len(self.free_memory) != len(self.allocated_memory):
                raise ValueError("Num GPUs changed! I don't feel so good.")
            
            self.free_memory -= self.allocated_memory
        except Exception as e:
            logging.exception(
                f"Failed to update free memory array because {e}."
            )

    def assign_gpus(
        self, 
        max_gpu_memory_mb: int = 14000,
        max_allocations: int = 1
    ) -> None:

        self.update_memory_array()

        if self.free_memory is None:
            raise ValueError("Free memory array is None, for some reason!")

        unallocated_processes = set(range(len(self.queued)))
        allocations_count = 0  # Initialize count of successful allocations

        while unallocated_processes:
            initial_unallocated_count = len(unallocated_processes)

            # Sort GPUs by their available memory in descending order
            sorted_gpus = sorted(enumerate(self.free_memory), key=lambda x: x[1], reverse=True)

            for process_index in unallocated_processes.copy():
                process = self.queued[process_index]
                total_memory_required_mb = process.tensorflow_memory_mb + process.cuda_overhead_mb

                for gpu_index, _ in sorted_gpus:
                    gpu_memory = self.free_memory[gpu_index]  # Local variable to track memory during allocation

                    # Check if adding this process would exceed the GPU's memory limit or the maximum allowed memory
                    if gpu_memory >= total_memory_required_mb and (max_allocations is None or allocations_count < max_allocations):
                        gpu_memory -= total_memory_required_mb
                        process.current_gpu = gpu_index
                        process.memory_assigned = total_memory_required_mb
                        self.allocated_memory[gpu_index] += total_memory_required_mb
                        unallocated_processes.remove(process_index)
                        allocations_count += 1  # Increment successful allocations
                        break

                    # Break out of the allocation process if the max allocations limit has been reached
                    if max_allocations is not None and allocations_count >= max_allocations:
                        break

                # Additional break to exit the outer loop when max allocations reached
                if max_allocations is not None and allocations_count >= max_allocations:
                    break

            if len(unallocated_processes) == initial_unallocated_count:
                break  # Break if no allocations were made in this pass
    
    def start_processes(self):

        self.assign_gpus()

        for command in self.queued:
            if command.current_gpu > -1 and self.queued and len(self.running) < self.max_num_concurent_processes:

                self.queued.remove(command)

                command.start()
                
                if command.process is not None:
                    self.running.append(command)
                elif not command.has_failed or not command.has_completed:
                    logging.info(
                        f"Attempting restart of {command.name}."
                    )
                    command.requeue()
                    time.sleep(self.process_start_wait_seconds)