import subprocess
import sys
import os
import shutil

def main():
    print("==================================================")
    print("Building VidChatBox Desktop Executable...")
    print("==================================================")
    
    # 1. Verify dependencies are installed
    print("Installing requirements from requirements.txt...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    print("Ensuring pywebview and pyinstaller are installed...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pywebview", "pyinstaller"], check=True)
    
    # 2. Prepare PyInstaller command
    if not os.path.exists("frontend"):
        print("Error: frontend folder not found!", file=sys.stderr)
        sys.exit(1)
        
    print("Packaging application with PyInstaller (One-File GUI Executable)...")
    
    # Build command
    # os.pathsep is ';' on Windows, ':' on Unix
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", "VidChatBox",
        "--add-data", f"frontend{os.pathsep}frontend",
        "--collect-all", "uvicorn",
        "--collect-all", "fastapi",
        "--collect-all", "anyio",
        "run_desktop.py"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n==================================================")
        print("[SUCCESS] VidChatBox packaged successfully!")
        print(f"Executable is located at: {os.path.abspath('dist/VidChatBox.exe')}")
        print("==================================================")
    else:
        print("\n==================================================")
        print("[ERROR] Packaging failed. See output above.")
        print("==================================================")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
