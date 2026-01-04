import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import torch
# Try the new SpeechBrain inference API first
try:
    from speechbrain.inference import SpeakerRecognition
except ImportError:
    # Fallback to the old API if the new one is not available
    from speechbrain.pretrained import SpeakerRecognition
from auth_utils import get_similarity
from spoof_detector import spoof_probability
from liveness_check import generate_challenge, verify_liveness
from deepfake_detector import detect_deepfake_audio
from log_manager import log_auth_attempt, save_user_embedding, get_user_embedding, get_all_users
from alert_system import trigger_security_alert
from explainability import get_spoof_explanation, get_deepfake_explanation
import os
import json
from typing import List, Dict, Any
import hashlib
import time

app = FastAPI(title="Live Voice Auth WebSocket")

print("Loading speaker embedding model (speechbrain ECAPA). This may take a few seconds...")

# Handle symlink issue on Windows by setting environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Try to load the model with better error handling
spkrec = None
try:
    print("Attempting to load model via torch.hub (recommended for Windows)...")
    # Use torch.hub.load to avoid HF hub symlink issues on Windows
    spkrec = torch.hub.load(
        'speechbrain/speechbrain', 'spkrec_ecapa_voxceleb',
        source='github', trust_repo=True,  # needed on newer torch
        run_opts={"device": "cpu"}, 
        savedir=os.path.join(os.path.dirname(__file__), "pretrained_models/spkrec_ecapa")
    )
    print("Model loaded successfully via torch.hub!")
except Exception as e:
    print(f"Error loading model via torch.hub: {e}")
    print("Falling back to direct model loading...")
    try:
        # Check if model files exist locally
        model_dir = os.path.join(os.path.dirname(__file__), "pretrained_models/spkrec_ecapa")
        required_files = [
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "classifier.ckpt",
            "label_encoder.txt"
        ]
        
        # Check if all required files exist
        all_files_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
        
        if os.path.exists(model_dir) and all_files_exist:
            print(f"Model files found in {model_dir}.")
            print("Attempting to load model (this may fail on Windows due to symlink issues)...")
            # Try to load the model with local_files_only parameter
            spkrec = SpeakerRecognition.from_hparams(
                source=model_dir,
                savedir=model_dir,
                run_opts={"device": "cpu"},
                use_auth_token=False,
                local_files_only=True  # Use only local files
            )
            print("Model loaded successfully!")
        else:
            print(f"Model files not found in {model_dir}. To use the real model, please:")
            print("1. Run the setup_model.py script to download the model files")
            print("2. Restart the server")
            print("Using dummy model for testing.")
            spkrec = None
    except Exception as e2:
        print(f"Error loading model via direct method: {e2}")
        print("This is a known issue on Windows with SpeechBrain symlink handling.")
        print("The system will use a dummy model for testing.")
        print("For production use, consider running on Linux or using a Docker container.")
        spkrec = None

# User session state management
user_sessions = {}  # {websocket_id: {user_id, registered_embedding, liveness_challenge, liveness_pending}}

def compute_log_hash(log_data: Dict) -> str:
    """
    Compute SHA256 hash of log entry for blockchain verification
    """
    # Create a sorted string representation of the log data
    log_string = json.dumps(log_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(log_string.encode('utf-8')).hexdigest()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_id = id(websocket)
    user_sessions[websocket_id] = {
        "user_id": None,
        "registered_embedding": None,
        "liveness_challenge": None,
        "liveness_pending": False,
        "liveness_passed": False
    }
    print(f"Client connected with session ID: {websocket_id}")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"Received {len(data)} bytes of audio data")
            # interpret incoming bytes as float32 PCM
            pcm = np.frombuffer(data, dtype=np.float32)
            print(f"PCM data shape: {pcm.shape}")
            if pcm.size < 160:  # sanity check
                print("Audio data too short")
                await websocket.send_text("ERR: chunk too short")
                continue

            # Get user session
            session = user_sessions[websocket_id]
            
            # If user_id not set, we need to initialize
            if session["user_id"] is None:
                # For demo purposes, we'll use a default user ID
                # In a real implementation, this would come from client
                session["user_id"] = "default_user"
                # Try to load existing embedding for this user
                embedding_data = get_user_embedding(session["user_id"])
                if embedding_data:
                    session["registered_embedding"] = np.frombuffer(embedding_data, dtype=np.float32)
                    print(f"Loaded existing embedding for user {session['user_id']}")

            # Deepfake detection (run first for security)
            is_deepfake, deepfake_confidence, _ = detect_deepfake_audio(pcm)
            print(f"Deepfake detection: is_deepfake={is_deepfake}, confidence={deepfake_confidence:.3f}")
            if is_deepfake and deepfake_confidence > 0.7:
                # Prepare log data
                log_data = {
                    "user_id": session["user_id"],
                    "timestamp": int(time.time()),
                    "result": "DEEPFAKE_DETECTED",
                    "deepfake_confidence": deepfake_confidence,
                    "audio_length": len(data)
                }
                
                # Compute hash for blockchain verification
                log_hash = compute_log_hash(log_data)
                
                # Log the attempt
                log_auth_attempt(
                    user_id=session["user_id"],
                    result="DEEPFAKE_DETECTED",
                    deepfake_confidence=deepfake_confidence,
                    audio_data=data
                )
                
                # Get deepfake explanation
                try:
                    explanation = get_deepfake_explanation(pcm, deepfake_confidence)
                    explanation_details = {
                        "explanations": explanation["explanations"],
                        "overall_confidence": explanation["overall_confidence"]
                    }
                except Exception as e:
                    print(f"Error getting deepfake explanation: {e}")
                    explanation_details = {"error": "Could not generate explanation"}
                
                # Trigger security alert for deepfake detection with explanation
                trigger_security_alert(
                    "DEEPFAKE_DETECTED",
                    {
                        "confidence": deepfake_confidence,
                        "audio_length": len(data),
                        "explanation": explanation_details,
                        "blockchain_hash": log_hash
                    },
                    session["user_id"]
                )
                
                # Send explanation to client
                explanation_msg = "DEEPFAKE_DETECTED: "
                if "explanations" in explanation_details:
                    deepfake_reasons: List[str] = []
                    for exp in explanation_details["explanations"]:
                        if isinstance(exp, dict) and "indicator" in exp:
                            deepfake_reasons.append(str(exp["indicator"]))
                    explanation_msg += ", ".join(deepfake_reasons[:3])  # Limit to first 3 reasons
                else:
                    explanation_msg += f"confidence={deepfake_confidence:.3f}"
                
                await websocket.send_text(explanation_msg)
                continue

            # If we're waiting for a liveness check response
            if session["liveness_pending"] and session["liveness_challenge"]:
                # For demo purposes, we'll simulate speech-to-text by using the challenge as the "spoken" text
                # In a real implementation, you would use an STT service here
                spoken_text = session["liveness_challenge"]  # Simulate perfect STT for demo
                is_valid, liveness_confidence, challenge = verify_liveness(spoken_text)
                
                if not is_valid:
                    # Prepare log data
                    log_data = {
                        "user_id": session["user_id"],
                        "timestamp": int(time.time()),
                        "result": "LIVENESS_FAILED",
                        "liveness_confidence": liveness_confidence,
                        "expected": challenge
                    }
                    
                    # Compute hash for blockchain verification
                    log_hash = compute_log_hash(log_data)
                    
                    # Log the attempt
                    log_auth_attempt(
                        user_id=session["user_id"],
                        result="LIVENESS_FAILED",
                        liveness_confidence=liveness_confidence,
                        audio_data=data
                    )
                    
                    # Trigger security alert for liveness failure
                    trigger_security_alert(
                        "LIVENESS_FAILED",
                        {
                            "confidence": liveness_confidence,
                            "expected": challenge,
                            "blockchain_hash": log_hash
                        },
                        session["user_id"]
                    )
                    
                    await websocket.send_text(f"LIVENESS_FAILED: confidence={liveness_confidence:.2f}, expected='{challenge}'")
                    session["liveness_pending"] = False
                    session["liveness_challenge"] = None
                    continue
                else:
                    # Prepare log data
                    log_data = {
                        "user_id": session["user_id"],
                        "timestamp": int(time.time()),
                        "result": "LIVENESS_PASSED",
                        "liveness_confidence": liveness_confidence
                    }
                    
                    # Compute hash for blockchain verification
                    log_hash = compute_log_hash(log_data)
                    
                    # Log the liveness pass
                    log_auth_attempt(
                        user_id=session["user_id"],
                        result="LIVENESS_PASSED",
                        liveness_confidence=liveness_confidence,
                        audio_data=data
                    )
                    await websocket.send_text(f"LIVENESS_PASSED: confidence={liveness_confidence:.2f}")
                    session["liveness_pending"] = False
                    session["liveness_challenge"] = None
                    session["liveness_passed"] = True
                    # Continue with normal authentication after liveness check

            # get embedding
            if spkrec is not None:
                waveform = torch.from_numpy(pcm).unsqueeze(0)  # [1, time]
                with torch.no_grad():
                    emb = spkrec.encode_batch(waveform).squeeze(0).cpu().numpy()
            else:
                # Dummy embedding for testing
                emb = np.random.rand(192).astype(np.float32)
            
            print(f"Generated embedding shape: {emb.shape}")

            # if first sample for this user, register user
            if session["registered_embedding"] is None:
                session["registered_embedding"] = emb
                # Save embedding to database
                save_user_embedding(session["user_id"], emb.tobytes())
                print(f"Registered new user embedding for {session['user_id']}")
                # Generate liveness challenge for next authentication
                session["liveness_challenge"] = generate_challenge()
                session["liveness_pending"] = True
                # Prepare log data
                log_data = {
                    "user_id": session["user_id"],
                    "timestamp": int(time.time()),
                    "result": "REGISTERED"
                }
                
                # Compute hash for blockchain verification
                log_hash = compute_log_hash(log_data)
                
                # Log the registration
                log_auth_attempt(
                    user_id=session["user_id"],
                    result="REGISTERED",
                    audio_data=data
                )
                await websocket.send_text(f"REGISTERED: voice sample stored for {session['user_id']}. Next time, say: '{session['liveness_challenge']}'")
                continue

            # quick anti-spoof heuristic (0..1)
            spf = spoof_probability(pcm)
            print(f"Spoof probability: {spf}")
            if spf > 0.5:
                # Prepare log data
                log_data = {
                    "user_id": session["user_id"],
                    "timestamp": int(time.time()),
                    "result": "SPOOF_DETECTED",
                    "spoof_probability": spf
                }
                
                # Compute hash for blockchain verification
                log_hash = compute_log_hash(log_data)
                
                # Log the attempt
                log_auth_attempt(
                    user_id=session["user_id"],
                    result="SPOOF_DETECTED",
                    spoof_probability=spf,
                    audio_data=data
                )
                
                # Get spoof explanation
                try:
                    explanation = get_spoof_explanation(pcm, spf)
                    explanation_details = {
                        "explanations": explanation["explanations"],
                        "overall_confidence": explanation["overall_confidence"]
                    }
                except Exception as e:
                    print(f"Error getting spoof explanation: {e}")
                    explanation_details = {"error": "Could not generate explanation"}
                
                # Trigger security alert for spoof detection with explanation
                trigger_security_alert(
                    "SPOOF_DETECTED",
                    {
                        "probability": spf,
                        "audio_length": len(data),
                        "explanation": explanation_details,
                        "blockchain_hash": log_hash
                    },
                    session["user_id"]
                )
                
                # Send explanation to client
                explanation_msg = "SPOOF_DETECTED: "
                if "explanations" in explanation_details:
                    spoof_reasons: List[str] = []
                    for exp in explanation_details["explanations"]:
                        if isinstance(exp, dict) and "indicator" in exp:
                            spoof_reasons.append(str(exp["indicator"]))
                    explanation_msg += ", ".join(spoof_reasons[:3])  # Limit to first 3 reasons
                else:
                    explanation_msg += f"prob={spf:.2f}"
                
                await websocket.send_text(explanation_msg)
                continue

            # For demo, we'll add a liveness check before authentication
            # In a real implementation, this would be more sophisticated
            if not session["liveness_passed"] and not session["liveness_pending"]:
                # Generate a new liveness challenge
                session["liveness_challenge"] = generate_challenge()
                session["liveness_pending"] = True
                await websocket.send_text(f"LIVENESS_CHALLENGE: Please say '{session['liveness_challenge']}'")
                continue

            # similarity
            sim = get_similarity(session["registered_embedding"], emb)
            print(f"Similarity: {sim}")
            if sim >= 0.70:
                # Prepare log data
                log_data = {
                    "user_id": session["user_id"],
                    "timestamp": int(time.time()),
                    "result": "ACCESS_GRANTED",
                    "similarity_score": sim,
                    "confidence_score": sim
                }
                
                # Compute hash for blockchain verification
                log_hash = compute_log_hash(log_data)
                
                # Log the successful authentication
                log_auth_attempt(
                    user_id=session["user_id"],
                    result="ACCESS_GRANTED",
                    similarity_score=sim,
                    confidence_score=sim,  # Using similarity as confidence for demo
                    audio_data=data
                )
                await websocket.send_text(f"ACCESS_GRANTED for {session['user_id']}: similarity={sim:.3f}")
            else:
                # Prepare log data
                log_data = {
                    "user_id": session["user_id"],
                    "timestamp": int(time.time()),
                    "result": "ACCESS_DENIED",
                    "similarity_score": sim,
                    "confidence_score": 1.0 - sim
                }
                
                # Compute hash for blockchain verification
                log_hash = compute_log_hash(log_data)
                
                # Log the failed authentication
                log_auth_attempt(
                    user_id=session["user_id"],
                    result="ACCESS_DENIED",
                    similarity_score=sim,
                    confidence_score=1.0 - sim,  # Inverse as confidence for demo
                    audio_data=data
                )
                await websocket.send_text(f"ACCESS_DENIED for {session['user_id']}: similarity={sim:.3f}")

    except WebSocketDisconnect:
        print(f"Client disconnected with session ID: {websocket_id}")
        # Clean up session
        if websocket_id in user_sessions:
            del user_sessions[websocket_id]
    except Exception as e:
        print("Server error:", e)
        try:
            await websocket.send_text(f"ERR: {e}")
        except:
            pass
        # Clean up session on error
        if websocket_id in user_sessions:
            del user_sessions[websocket_id]

# Add endpoint for retrieving logs
@app.get("/logs")
async def get_logs(limit: int = 100):
    """
    Retrieve authentication logs
    """
    from log_manager import get_auth_logs
    return get_auth_logs(limit)

@app.get("/logs/count")
async def get_log_count():
    """
    Get total number of log entries
    """
    from log_manager import get_log_count
    return {"count": get_log_count()}

@app.get("/users")
async def get_users():
    """
    Get list of all registered users
    """
    return {"users": get_all_users()}

@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """
    Get recent security alerts
    """
    from alert_system import get_recent_security_alerts
    return get_recent_security_alerts(limit)

@app.get("/alerts/count")
async def get_alert_count():
    """
    Get total number of security alerts
    """
    from alert_system import get_alert_count
    return {"count": get_alert_count()}

# Serve the professional UI HTML file
@app.get("/")
async def professional_ui():
    """
    Serve the professional UI
    """
    import os
    from fastapi.responses import HTMLResponse
    ui_path = os.path.join(os.path.dirname(__file__), "professional_ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r") as f:
            content = f.read()
        return HTMLResponse(content=content, headers={"Content-Type": "text/html; charset=utf-8"})
    else:
        return HTMLResponse(content="<h1>Professional UI not found</h1><p>Please run the system to generate the UI.</p>", headers={"Content-Type": "text/html; charset=utf-8"})

# Serve the dashboard HTML file
@app.get("/dashboard")
async def dashboard():
    """
    Serve the admin dashboard
    """
    import os
    from fastapi.responses import HTMLResponse
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r") as f:
            content = f.read()
        return HTMLResponse(content=content, headers={"Content-Type": "text/html; charset=utf-8"})
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1><p>Run the system to generate the dashboard.</p>", headers={"Content-Type": "text/html; charset=utf-8"})