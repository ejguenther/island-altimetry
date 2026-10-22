"""Lets scripts import sibling scripts (e.g. acquire) as modules."""
import importlib.util
from pathlib import Path


def load_acquire():
    spec = importlib.util.spec_from_file_location("acquire", Path(__file__).with_name("acquire.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_train():
    spec = importlib.util.spec_from_file_location("train_coastal_rf", Path(__file__).with_name("train_coastal_rf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
