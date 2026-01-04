import uvicorn
import os

if __name__ == "__main__":
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting Voice Authentication Server...")
    print("Server will be available at http://localhost:8001")
    print("WebSocket endpoint: ws://localhost:8001/ws")
    
    # Import the app from main.py
    from main import app
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")