"""
Automatic version tracking for gpu-tigramite
Generates unique build identifiers to prevent multi-version confusion
"""

import os
import subprocess
from datetime import datetime

# Base version (update manually for releases)
BASE_VERSION = "1.0.0"

def get_git_hash():
    """Get short git commit hash if available"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=1,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None

def get_version():
    """
    Generate version string with build metadata.
    
    Format: 1.0.0+build.YYYYMMDD.HHMMSS[.git.HASH]
    
    Examples:
        1.0.0+build.20251121.134523
        1.0.0+build.20251121.134523.git.a3f2b1c
    
    Build metadata is ignored by pip for dependency resolution,
    but provides unique identification for debugging.
    """
    # Get build timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d.%H%M%S")
    
    # Start with base version + build timestamp
    version = f"{BASE_VERSION}+build.{timestamp}"
    
    # Add git hash if available
    git_hash = get_git_hash()
    if git_hash:
        version += f".git.{git_hash}"
    
    return version

# Generate version at import time
__version__ = get_version()

if __name__ == '__main__':
    print(__version__)