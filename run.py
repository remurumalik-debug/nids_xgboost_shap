"""
Launcher for the Explainable NIDS (Flask + Vanilla JS version)
Run this script to start the web application.
"""
import os
import sys
import subprocess

def main():
    backend_dir = os.path.join(os.path.dirname(__file__), "flask_backend")
    
    # Check if model exists, if not train it
    model_path = os.path.join(backend_dir, "model", "xgb_model.json")
    if not os.path.exists(model_path):
        print("Model not found. Training...")
        result = subprocess.run([sys.executable, "train.py"], cwd=backend_dir)
        if result.returncode != 0:
            print("Training failed!")
            return 1
    
    # Start Flask app
    print("Starting Flask server on http://localhost:5050")
    print("Press CTRL+C to stop")
    os.chdir(backend_dir)
    os.execv(sys.executable, [sys.executable, "app.py"])

if __name__ == "__main__":
    main()