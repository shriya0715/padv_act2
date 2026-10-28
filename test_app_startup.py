"""
Streamlit App Integration Test
Verifies that the dashboard starts up, binds to localhost, and returns HTTP 200.
"""

import subprocess
import time
import urllib.request
import sys

def test_startup():
    print("Starting Streamlit app server...")
    cmd = [
        sys.executable,
        "-m", "streamlit", "run", "app.py",
        "--server.port", "8502",
        "--server.headless", "true"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    url = "http://localhost:8502"
    success = False
    
    for attempt in range(1, 25):
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print(f"--> SUCCESS: Streamlit app responded with HTTP {response.status} on attempt {attempt}!")
                    success = True
                    break
        except Exception as e:
            # Server still warming up
            if attempt % 5 == 0:
                print(f"Attempt {attempt}: waiting for server (last error: {e})")

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()

    if not success:
        print("--> FAILED: Server did not respond within expected timeout.")
        out, err = proc.communicate()
        print("STDOUT:", out)
        print("STDERR:", err)
        sys.exit(1)
    else:
        print("--> App startup verification PASSED.")

if __name__ == "__main__":
    test_startup()
