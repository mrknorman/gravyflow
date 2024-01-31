import logging
import time
from pathlib import Path
from typing import List
import datetime
import json
import gravyflow as gf
import pytz

# Configuring the logger
logging.basicConfig(filename='gpu_monitor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def format_current_datetime():
    """Returns the current datetime in UK timezone as a formatted string."""
    uk_timezone = pytz.timezone('Europe/London')
    uk_time = datetime.datetime.now(uk_timezone)
    return uk_time.strftime("%Y-%m-%d %H:%M:%S")

def load_config(
        path : Path
    ):
    """Loads configuration parameters from a JSON file."""
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

def format_gpu_info_table(gpu_memory, gpu_utilization):
    """Formats the GPU information into a table-like string."""
    header = f"{'GPU':<10}{'Memory (MB)':<20}{'Utilization (%)':<20}\n"
    divider = "-" * 50 + "\n"
    rows = ""

    for i, (memory, utilization) in enumerate(zip(gpu_memory, gpu_utilization)):
        row = f"|{f'GPU {i}':<10}|{memory:<20}|{utilization:<20}|\n"
        rows += row

    return header + divider + rows

def compose_message(
        num_free : int, 
        gpu_memory : List[float], 
        gpu_utilization : List[float], 
        current_datetime : datetime.datetime, 
        thresholds : List[float] 
    ) -> str:
    """Composes the alert email message based on the GPU status."""
    min_alert_threshold, empty_cluster_threshold = thresholds
    if num_free > min_alert_threshold:
        if num_free > empty_cluster_threshold:
            message = (
                f"The cluster is nearly empty with {num_free} free GPUs at HANFORD "
                f"as of {current_datetime}!\n\n"
            )
        else:
            message = (
                f"There are more free GPUs available! Total free GPUs: {num_free} "
                f"at HANFORD as of {current_datetime}!\n\n"
            )
    else:
        if num_free <= empty_cluster_threshold:
            message = (
                f"The cluster is getting busier! Fewer than {empty_cluster_threshold} " 
                f"GPUs free at HANFORD as of {current_datetime}!\n\n"
            )
        else:
            message = (
                f"GPU availability has decreased. Total free GPUs: {num_free} "
                f" at HANFORD as of {current_datetime}!\n\n"
            )

    message += format_gpu_info_table(gpu_memory, gpu_utilization)
    return message

def check_and_send_alert(last_num_free, last_check_time, config):
    """Checks the GPU status and sends an alert if needed."""
    current_datetime = format_current_datetime()
    
    try:
        gpu_memory = gf.get_memory_array()
        gpu_utilization = gf.get_gpu_utilization_array()
    except Exception as e:
        logging.error(f"Error fetching GPU data: {e}")
        return last_num_free, last_check_time

    num_free = sum((gpu_memory > config['min_memory_mb']) * (
        gpu_utilization < config['max_utilization_percentage'])
    )

    thresholds = (config['min_alert_threshold'], config['empty_cluster_threshold'])
    alert_condition = (
        (last_num_free <= thresholds[0] and num_free > thresholds[0]) or
        (last_num_free > thresholds[0] and num_free <= thresholds[0]) or
        (last_num_free <= thresholds[1] and num_free > thresholds[1]) or
        (last_num_free > thresholds[1] and num_free <= thresholds[1])
    )

    if alert_condition and (time.time() - last_check_time >= config['alert_interval_seconds']):
        message = compose_message(
            num_free,
            gpu_memory, 
            gpu_utilization, 
            current_datetime, 
            thresholds
        )
        gf.send_email(
            "GPU Update", 
            message, 
            config['recipient_email'], 
            Path(config['email_config_path'])
        )
        logging.info("Alert email sent.")
        return num_free, time.time()
    
    return last_num_free, last_check_time

if __name__ == "__main__":
    config_path = "./gravyflow/alert_settings.json"  # Path to your JSON config file
    config = load_config(config_path)

    last_num_free = 0
    last_check_time = 0

    while True:
        last_num_free, last_check_time = check_and_send_alert(
            last_num_free, last_check_time, config
        )
        time.sleep(config['check_interval_seconds'])
        logging.info("Checking GPU status...")
