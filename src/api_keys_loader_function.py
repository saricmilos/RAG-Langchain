import os
from getpass import getpass

def get_environment_variable(key: str, save_if_missing: bool = True) -> str:
    """
    Retrieve an environment variable safely. 

    Parameters:
    - key (str): Name of the environment variable to retrieve.
    - save_if_missing (bool): If True, stores user input in os.environ if the variable is missing.

    Returns:
    - str: The value of the environment variable.
    """
    value = os.environ.get(key)
    
    if value:
        print(f"\n[INFO]: '{key}' retrieved successfully from environment variables.")
    else:
        print(f"\n[WARNING]: '{key}' not found in environment variables.")
        value = getpass(f"Please enter your '{key}': ")
        
        # Optionally save it to the environment for this session
        if save_if_missing:
            os.environ[key] = value
            print(f"[INFO]: '{key}' has been set for this session.")

    return value