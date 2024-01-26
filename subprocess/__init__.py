import subprocess
import time
import threading

def run_command(command):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

    def stdout_callback(process, s):
        print('Stdout: ' + s)
    def stderr_callback(process, s):
        print('Stderr: ' + s)

    stdout_thread = threading.Thread(target=_read_stream, args=(process.stdout, process, stdout_callback))
    stderr_thread = threading.Thread(target=_read_stream, args=(process.stderr, process, stderr_callback))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.returncode

def run_command_iterator(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()

        if process.poll() is not None and not output and not error:
            break

        if output:
            yield f"Output: {output.strip()}"
        if error:
            yield f"Error: {error.strip()}"

        time.sleep(0.1)

    yield f"Return: {process.returncode}" 


def _read_stream(stream, process, callback=None):
    for line in iter(stream.readline, b''):
        if not line:
            return
        if callback is not None:
            callback(process, line.strip())