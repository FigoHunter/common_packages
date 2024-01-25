import subprocess
import time

def run_command(command, callback=None):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()

        if process.poll() is not None and not output and not error:
            break

        if output:
            print(f"Output: {output.strip()}")
            if callback is not None:
                callback(process, output.strip())
        if error:
            print(f"Error: {error.strip()}")

        time.sleep(0.1)

    return process.returncode

def run_command_iterator(command, callback=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()

        if process.poll() is not None and not output and not error:
            break

        if output:
            yield f"Output: {output.strip()}"
            if callback is not None:
                callback(process, output.strip())
        if error:
            yield f"Error: {error.strip()}"

        time.sleep(0.1)

    yield f"Return: {process.returncode}" 