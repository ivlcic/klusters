import os
import sys
import locale
import io
from dotenv import load_dotenv
# Set the locale to UTF-8
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Override the default encoding for standard input/output
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure the environment uses UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

# Determine the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the appropriate subdirectory for virtual environment binaries
if os.name == 'nt':  # Windows
    venv_bin_dir = os.path.join(current_script_dir, 'venv', 'Scripts')
else:  # Other operating systems (Linux, macOS, etc.)
    venv_bin_dir = os.path.join(current_script_dir, 'venv', 'bin')

# Update the PATH environment variable
os.environ['PATH'] = venv_bin_dir + os.pathsep + os.environ['PATH']
os.environ['CURR_DIR'] = current_script_dir
load_dotenv()
