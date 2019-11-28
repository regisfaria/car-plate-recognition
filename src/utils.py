'''
Here i'll be writting utility functions if needed

author: Régis Faria
email: regisprogramming@gmail.com
'''
import os
import time
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

