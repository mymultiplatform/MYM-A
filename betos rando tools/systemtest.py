import psutil
import os

def print_system_info():
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)
    
    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Total Memory: {memory_gb:.1f} GB")
    print(f"Memory Usage: {memory.percent}%")

# Add to main:
print_system_info()
