import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Dict, Any



@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """reads raml file and returns

    Args: path_to_yaml(str): path like input

    Raises:
        ValueError: if the yaml is not valid
        e: empty file

    Returns: ConfigBox: config box type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is not valid")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    
    Args:
        path_list (list): list of path of directories
        verbose (bool): ignore if multiple dirs is to be created
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)  # FIXED: was os.path.makedirs
        if verbose:
            logger.info(f"created directory: {path}")

#def create_directories(path_to_directories: list, verbase=True):
#   """create a list odf directories
#    Args:
#       path_to_directories(list): list of paths like input
#       ignore_log(bool, optional): ignore if multiple dirs is to be created. Defaults to False.
#  """
#    for path in path_to_directories:
#       os.makedirs(path, exist_ok=True)
#       if verbose:
#            logger.info(f"created directory: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
    Args:
        path(Path): path of the file
        
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} KB"        
        