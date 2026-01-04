
import asyncio
import websockets
import numpy as np
import argparse
import sys
import threading
import queue

# Try to import sounddevice
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("Warning: sounddevice not found. Using simulated audio.")

SAMPLE_RATE = 16000

class VoiceClient:
    def __init__(self, url="ws://localhost:8001/ws", user_id="default_user"):
        self.url = url
        self.user_id = user_id
        self.running = False
        self.input_queue = queue.Queue()
        
    async def connect(self):
        print(f"Connecting to {self.url}...")
        try:
            async with websockets.connect(self.url) as websocket:
                print("Connected!")
                print("Options:")
                print("  [r] Send 'Human' Voice (Registration/Auth)")
                print("  [n] Send 'Noise' (Deepfake Test)")
                print("  [t] Send 'Tone' (Spoof Test)")
                print("  [m] Microphone Input (if available)")
                print("  [q] Quit")
                
                self.running = True
                
                # Start input thread
                input_thread = threading.Thread(target=self.input_loop)
                input_thread.daemon = True
                input_thread.start()
                
                # Main loop
                send_task = asyncio.create_task(self.send_loop(websocket))
                recv_task = asyncio.create_task(self.receive_loop(websocket))
                
                await asyncio.gather(send_task, recv_task)
                
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.running = False

    def input_loop(self):
        while self.running:
            cmd = input().strip().lower()
            if cmd:
                self.input_queue.put(cmd)
            if cmd == 'q':
                break

    def generate_audio(self, pattern, duration=2.0):
        chunk_size = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, chunk_size)
        
        if pattern == "human":
            # Simulate human voice characteristics
            f0 = 150 + 10 * np.sin(2 * np.pi * 3 * t) 
            phase = np.cumsum(2 * np.pi * f0 / SAMPLE_RATE)
            signal = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
            envelope = 0.5 + 0.2 * np.sin(2 * np.pi * 2 * t)
            data = (signal * envelope).astype(np.float32)
            data += np.random.normal(0, 0.005, chunk_size).astype(np.float32)
            return data
            
        elif pattern == "noise":
            return np.random.randn(chunk_size).astype(np.float32) * 0.1
            
        elif pattern == "tone":
            return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
            
        return np.zeros(chunk_size, dtype=np.float32)

    async def record_microphone(self, duration=3.0):
        if not HAS_AUDIO:
            print("Microphone not available.")
            return None
            
        print(f"Recording for {duration} seconds...")
        try:
            recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            return recording.flatten()
        except Exception as e:
            print(f"Recording error: {e}")
            return None

    async def send_loop(self, websocket):
        while self.running:
            try:
                # Non-blocking get
                try:
                    cmd = self.input_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                if cmd == 'q':
                    self.running = False
                    await websocket.close()
                    break
                    
                data = None
                if cmd == 'r':
                    print("Sending Human Voice Simulation...")
                    data = self.generate_audio("human")
                elif cmd == 'n':
                    print("Sending Noise...")
                    data = self.generate_audio("noise")
                elif cmd == 't':
                    print("Sending Tone...")
                    data = self.generate_audio("tone")
                elif cmd == 'm':
                    data = await self.record_microphone()
                
                if data is not None:
                    await websocket.send(data.tobytes())
                    # If liveness challenge might come, allow sending again
                    # But for now, user manually triggers response
                
            except Exception as e:
                print(f"Send error: {e}")
                break

    async def receive_loop(self, websocket):
        while self.running:
            try:
                message = await websocket.recv()
                print(f"\nSERVER: {message}")
                if "LIVENESS_CHALLENGE" in str(message):
                    print(">>> CHALLENGE RECEIVED! Type 'r' or 'm' to send response audio <<<")
            except websockets.exceptions.ConnectionClosed:
                print("Server closed connection")
                self.running = False
                break
            except Exception as e:
                print(f"Receive error: {e}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Auth Client")
    parser.add_argument("--url", default="ws://localhost:8001/ws", help="WebSocket URL")
    args = parser.parse_args()
    
    client = VoiceClient(url=args.url)
    try:
        asyncio.run(client.connect())
    except KeyboardInterrupt:
        print("\nExiting...")
