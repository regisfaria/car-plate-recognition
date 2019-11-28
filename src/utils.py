'''
Here i'll be writting utility functions if needed

@AUTHOR: Régis Faria
@EMAIL: regisprogramming@gmail.com
'''
import os
import time
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

