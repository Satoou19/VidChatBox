import os
import sys
import socket
import threading
import time
import base64
import uvicorn
import webview

# Ensure backend directory is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.main import app

import json

class DesktopAPI:
    def __init__(self):
        self.window = None
        # Setup settings file path in APPDATA
        appdata = os.getenv("APPDATA") or os.path.expanduser("~")
        self.settings_dir = os.path.join(appdata, "VidChatBox")
        os.makedirs(self.settings_dir, exist_ok=True)
        self.settings_path = os.path.join(self.settings_dir, "settings.json")

    def set_window(self, window):
        self.window = window

    def save_settings(self, settings_dict):
        """Saves settings dict to settings.json in user APPDATA folder."""
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(settings_dict, f, ensure_ascii=False, indent=2)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_settings(self):
        """Loads settings dict from settings.json in user APPDATA folder."""
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    return {"success": True, "settings": json.load(f)}
            return {"success": True, "settings": {}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_file_dialog(self, base64_content, filename):
        """Opens a native save file dialog and writes the decoded base64 content to it."""
        try:
            if not self.window:
                return {"success": False, "error": "Desktop window not initialized."}
            
            # Show save file dialog
            file_path = self.window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=filename
            )
            
            if file_path:
                # Handle cases where create_file_dialog returns a tuple or list of strings
                if isinstance(file_path, (list, tuple)):
                    if len(file_path) > 0:
                        file_path = file_path[0]
                    else:
                        return {"success": False, "error": "Save operation was cancelled by user."}
                
                # Strip base64 header if present (e.g. data:text/markdown;base64,...)
                if "," in base64_content:
                    base64_content = base64_content.split(",", 1)[1]
                
                # Decode and save
                content_bytes = base64.b64decode(base64_content)
                with open(file_path, "wb") as f:
                    f.write(content_bytes)
                
                return {"success": True, "path": file_path}
            
            return {"success": False, "error": "Save operation was cancelled by user."}
        except Exception as e:
            return {"success": False, "error": f"Failed to save file: {str(e)}"}

def get_free_port():
    """Finds a free TCP port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def wait_for_port(port, timeout=5.0):
    """Waits for the local server to start listening on the given port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.1)
    return False

def run_server(port):
    """Starts the FastAPI backend using Uvicorn."""
    # Run with warning log level to keep console quiet
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

def main():
    port = get_free_port()
    
    # Start FastAPI in a background daemon thread
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()
    
    # Wait for the local server to be ready before opening webview
    if not wait_for_port(port):
        print(f"Error: FastAPI server did not start on port {port} in time.", file=sys.stderr)
        sys.exit(1)
        
    # Instantiate JS API
    api = DesktopAPI()

    # Launch pywebview window
    window = webview.create_window(
        "VidChatBox",
        f"http://127.0.0.1:{port}",
        width=1280,
        height=800,
        min_size=(1024, 768),
        js_api=api
    )
    api.set_window(window)

    # Configure persistent storage path for user data (like localStorage and cookies)
    appdata = os.getenv("APPDATA") or os.path.expanduser("~")
    storage_path = os.path.join(appdata, "VidChatBox", "web_cache")
    os.makedirs(storage_path, exist_ok=True)
    
    webview.start(private_mode=False, storage_path=storage_path)

if __name__ == "__main__":
    main()
