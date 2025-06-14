import json
import os
import psutil
import hashlib

def nameit(input_file, replacement, **kwargs):
    base_name = os.path.basename(input_file).rsplit('.zarr', 1)[0]
    parts = base_name.split('-', 1)
    modified_name = f"{replacement}-{parts[1]}" if len(parts) > 1 else replacement

    for key, value in kwargs.items():
        modified_name += f"-{key}{value}"

    modified_name += '.zarr'
    return modified_name

def monitor_cpu_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = process.cpu_percent(interval=1)
    cpu_count = len(process.cpu_affinity())
    return cpu_usage, cpu_count

def config_hash(section: dict):
    string = json.dumps(section, sort_keys=True).encode()
    return hashlib.md5(string).hexdigest()[:8]

def handle_output(dask_array, output_path, save_output=True):
    dask_array.to_zarr(output_path, overwrite=True)
    if not save_output:
        import shutil
        shutil.rmtree(output_path, ignore_errors=True)
    return output_path
