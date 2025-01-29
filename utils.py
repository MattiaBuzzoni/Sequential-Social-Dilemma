import os
import socket
import datetime
import psutil
import numpy as np


def get_name_weights(idx):
    file_name_str = 'weights_agent_'

    return (os.path.abspath(os.path.dirname(__file__)) + '/weights/'
            + file_name_str + '_' + str(idx) + '.keras')


def report_training_status(trainer):
    """
    Generates and prints the output summary of the training metrics and system information.
    """
    # Collect system information and training metrics
    result = {
        "System Information": {
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
            "Hostname": socket.gethostname(),
            "CPU Utilization (%)": f"{psutil.cpu_percent()}%",
            "CPU Utilization per Core (%)": ', '.join([f"{x}%" for x in psutil.cpu_percent(percpu=True)]),
            "RAM Total (GB)": f"{psutil.virtual_memory().total / (1024 ** 3):.2f}",
            "RAM Available (GB)": f"{psutil.virtual_memory().available / (1024 ** 3):.2f}",
            "Swap Used (GB)": f"{psutil.swap_memory().used / (1024 ** 3):.2f}",
            "Disk Read (GB)": f"{psutil.disk_io_counters().read_bytes / (1024 ** 3):.2f}",
            "Disk Write (GB)": f"{psutil.disk_io_counters().write_bytes / (1024 ** 3):.2f}",
        },
        "Training Metrics": {
            "Episode Reward Max": np.max(np.max(trainer.reward_array, axis=(0, 1))),
            "Episode Reward Min": np.min(np.min(trainer.reward_array, axis=(0, 1))),
            "Episode Reward Mean": round(np.mean(np.mean(trainer.reward_array, axis=(0, 1))), 1),
            "Total Episodes": trainer.episodes_number,
            "Steps This Iteration": trainer.max_ts,
        }
    }

    # Print the result summary
    print("\nTraining results:")
    for section, metrics in result.items():
        print(f"\n{section}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    # Print a separator line for clarity
    print("\n" + "-" * 40)

    # Return the result for further use
    return result
