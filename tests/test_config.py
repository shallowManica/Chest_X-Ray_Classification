"""Tests for configuration and path security."""
import os
from pathlib import Path
import pytest


def test_config_imports():
    """Test that config module imports correctly."""
    from chest_x_ray_classification.config import (
        RAW_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        PROCESSED_DATA_DIR,
        PROJECT_DIR
    )
    
    # All paths should be Path objects
    assert isinstance(RAW_DATA_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
    assert isinstance(FIGURES_DIR, Path)
    assert isinstance(PROCESSED_DATA_DIR, Path)
    assert isinstance(PROJECT_DIR, Path)


def test_no_hardcoded_personal_paths():
    """Verify that no personal paths are hardcoded in Python files."""
    project_root = Path(__file__).resolve().parents[1]
    python_files = list(project_root.glob("chest_x_ray_classification/**/*.py"))
    
    # Check for Windows absolute paths that might contain personal info
    # Using partial patterns to avoid including full paths in the test itself
    personal_patterns = [
        r"[A-Z]:/",  # Windows drive letter paths like C:/ or H:/
        r"[A-Z]:\\",  # Windows drive letter paths (escaped) like C:\\ or H:\\
    ]
    
    import re
    issues_found = []
    
    for py_file in python_files:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for pattern in personal_patterns:
                # Look for assignment statements with these patterns
                # This regex looks for variable assignments containing drive paths
                assignment_pattern = rf'\w+\s*=\s*["\'].*{pattern}.*["\']'
                matches = re.findall(assignment_pattern, content)
                if matches:
                    issues_found.append(
                        f"Found potential personal path in {py_file}: {matches[0][:50]}..."
                    )
    
    assert len(issues_found) == 0, (
        f"Found {len(issues_found)} potential personal paths:\n" + 
        "\n".join(issues_found)
    )


def test_config_uses_relative_paths_by_default():
    """Test that config uses relative paths when no env vars are set."""
    # Save current env vars
    env_vars_to_check = ['DATA_DIR', 'MODELS_DIR', 'PROCESSED_DATA_DIR', 'FIGURES_DIR']
    old_env_vars = {var: os.environ.get(var) for var in env_vars_to_check}
    
    try:
        # Clear all config-related env vars
        for var in env_vars_to_check:
            if var in os.environ:
                del os.environ[var]
        
        # Reload config
        import importlib
        import chest_x_ray_classification.config as config
        importlib.reload(config)
        
        # Paths should be relative to project
        assert 'data' in str(config.RAW_DATA_DIR).lower()
        assert 'models' in str(config.MODELS_DIR).lower()
        
    finally:
        # Restore all env vars
        for var, value in old_env_vars.items():
            if value:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]


def test_config_respects_environment_variables():
    """Test that config respects environment variables when set."""
    test_path = "/custom/test/path"
    
    # Save current env var
    old_data_dir = os.environ.get('DATA_DIR')
    
    try:
        # Set custom env var
        os.environ['DATA_DIR'] = test_path
        
        # Reload config
        import importlib
        import chest_x_ray_classification.config as config
        importlib.reload(config)
        
        # Should use custom path
        assert str(config.RAW_DATA_DIR) == test_path
        
    finally:
        # Restore env var
        if old_data_dir:
            os.environ['DATA_DIR'] = old_data_dir
        elif 'DATA_DIR' in os.environ:
            del os.environ['DATA_DIR']
