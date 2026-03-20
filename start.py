

import os
import subprocess

port = os.environ.get("PORT", "8501")

# Remove the broken env var so streamlit doesn't read it
env = os.environ.copy()
env.pop("STREAMLIT_SERVER_PORT", None)

subprocess.run([
    "streamlit", "run", "app.py",
    f"--server.port={port}",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--server.runOnSave=false"
], env=env)