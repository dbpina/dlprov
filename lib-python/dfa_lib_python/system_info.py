import os
import platform
import socket
import psutil
import shutil
import subprocess
from datetime import datetime
import uuid

def is_container():
    try:
        with open("/proc/1/cgroup", "rt") as f:
            content = f.read()
            return "docker" in content or "kubepods" in content or "containerd" in content
    except Exception:
        return False

def get_memory_limit():
    try:
        if os.path.exists("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                limit = int(f.read())
                if limit > 1 << 50:
                    return None
                return round(limit / (1024 ** 3), 2)
    except Exception:
        pass
    return None

def get_gpu_info():
    try:
        gpu_info = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"
        ]).decode().strip()
        gpus = [line.strip().split(',')[0] for line in gpu_info.splitlines()]
        return ", ".join(gpus)
    except Exception:
        return "Not available"

def get_system_info():
    in_container = is_container()
    os_name = platform.system()
    os_version = platform.version()
    platform_desc = platform.platform()
    hostname = socket.gethostname()
    arch = platform.machine()
    processor = platform.processor()

    if os_name == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        processor = line.split(":")[1].strip()
                        break
        except Exception:
            pass
    elif os_name == "Darwin":
        try:
            processor = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
        except Exception:
            pass

    ram_total = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    ram_limit = get_memory_limit()
    disk_total, disk_used, disk_free = shutil.disk_usage("/")
    disk_total_gb = round(disk_total / (1024 ** 3), 2)
    disk_used_gb = round(disk_used / (1024 ** 3), 2)
    disk_free_gb = round(disk_free / (1024 ** 3), 2)
    gpus = get_gpu_info()

    return {
        "hostname": hostname,
        "os": os_name,
        "os_version": os_version,
        "platform": platform_desc,
        "architecture": arch,
        "processor": processor,
        "ram_total_gb": ram_total,
        "ram_limit_gb": ram_limit,
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "disk_free_gb": disk_free_gb,
        "gpus": gpus,
        "in_container": in_container
    }

# Example usage:
# if __name__ == "__main__":
#     info = get_system_info_for_db()
#     for k, v in info.items():
#         print(f"{k}: {v}")
