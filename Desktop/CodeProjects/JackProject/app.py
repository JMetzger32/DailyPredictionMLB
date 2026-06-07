"""
app.py — thin wrapper so Render's "gunicorn app:app" still works
after files were reorganized into Main/, updates/, Databases_and_logs/.
"""
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "updates"))
sys.path.insert(0, os.path.join(_here, "Main"))

# Import the Flask app object from Main/app.py.
# Using importlib avoids the naming conflict with this file.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("main_app", os.path.join(_here, "Main", "app.py"))
_mod  = _ilu.module_from_spec(_spec)
sys.modules["main_app"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app
