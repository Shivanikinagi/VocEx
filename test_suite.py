
import asyncio
import websockets
import numpy as np
import json
import time
import math

async def send_audio_chunk(websocket, chunk_size=32000, pattern="silence"): # 2 seconds
    """
    Send an audio chunk to the websocket
    pattern: 'silence', 'noise', 'tone', 'human_sim'
    """
    sr = 16000
    t = np.linspace(0, chunk_size/sr, chunk_size)
    
    if pattern == "silence":
        data = np.zeros(chunk_size, dtype=np.float32)
    elif pattern == "noise":
        data = np.random.randn(chunk_size).astype(np.float32) * 0.1
    elif pattern == "tone":
        # Pure tone - triggers deepfake (stable pitch)
        data = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    elif pattern == "chirp":
        data = np.sin(2 * np.pi * 440 * t * t).astype(np.float32) * 0.5
    elif pattern == "human_sim":
        # Simulate human voice characteristics:
        # 1. Fundamental frequency varying (Pitch variance)
        f0 = 150 + 10 * np.sin(2 * np.pi * 3 * t) # Varying around 150Hz
        phase = np.cumsum(2 * np.pi * f0 / sr)
        
        # 2. Harmonics (rich spectrum, low flatness)
        signal = 0.5 * np.sin(phase) + \
                 0.3 * np.sin(2 * phase) + \
                 0.2 * np.sin(3 * phase)
        
        # 3. Amplitude modulation (Energy variance)
        envelope = 0.5 + 0.2 * np.sin(2 * np.pi * 2 * t)
        
        data = (signal * envelope).astype(np.float32)
        
        # 4. Add slight noise
        data += np.random.normal(0, 0.005, chunk_size).astype(np.float32)
        
    else:
        data = np.zeros(chunk_size, dtype=np.float32)
    
    print(f"Sending {pattern} data ({len(data)} samples)...")
    await websocket.send(data.tobytes())

async def run_test_sample(sample_id, description, pattern, timeout=10.0):
    uri = "ws://localhost:8001/ws"
    print(f"\n--- Running Sample {sample_id}: {description} ---")
    try:
        async with websockets.connect(uri) as websocket:
            # Send audio
            await send_audio_chunk(websocket, pattern=pattern, chunk_size=32000)
            
            # Receive response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                print(f"Response: {response}")
                
                # If we get a liveness challenge, send another chunk
                if "LIVENESS_CHALLENGE" in str(response):
                    print("Received Liveness Challenge. Sending response audio...")
                    await send_audio_chunk(websocket, pattern=pattern, chunk_size=48000) # 3s response
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    print(f"Response after Liveness: {response}")
                    
                    # Check for Auth result
                    if "LIVENESS_PASSED" in str(response):
                         response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                         print(f"Final Auth Response: {response}")

            except asyncio.TimeoutError:
                print("Timeout waiting for response")
                
    except Exception as e:
        print(f"Connection failed: {e}")

async def main():
    await asyncio.sleep(1)
    
    samples = [
        # 1. Registration (Human Sim - Should pass Deepfake and Register)
        (1, "Registration (Human Sim)", "human_sim"),
        
        # 2. Auth Success (Same Human Sim - Should Match)
        (2, "Authentication Success (Human Sim)", "human_sim"),
        
        # 3. Deepfake Attempt (White Noise)
        (3, "Deepfake Attempt (White Noise)", "noise"),
        
        # 4. Spoof Attempt (Pure Tone)
        (4, "Spoof/Deepfake Check (Pure Tone)", "tone"),
        
        # 5. Liveness Check (Human Sim again)
        (5, "Liveness Flow", "human_sim"),
        
        # 6-10 Randoms
        (6, "Deepfake Re-test", "noise"),
        (7, "Chirp Test", "chirp"),
        (8, "Human Sim Var 1", "human_sim"),
        (9, "Human Sim Var 2", "human_sim"),
        (10, "Silence", "silence")
    ]
    
    for sid, desc, pat in samples:
        await run_test_sample(sid, desc, pat)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
