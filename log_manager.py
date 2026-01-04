import sqlite3
import json
import base64
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class LogManager:
    """
    Manages authentication logs with SQLite database
    """
    
    def __init__(self, db_path: str = "auth_logs.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """
        Initialize the database with the required table
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                result TEXT NOT NULL,
                confidence_score REAL,
                similarity_score REAL,
                spoof_probability REAL,
                deepfake_confidence REAL,
                liveness_confidence REAL,
                audio_data BLOB,
                metadata TEXT
            )
        ''')
        
        # Create users table for storing voice embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_attempt(self, user_id: Optional[str] = None, result: Optional[str] = None, confidence_score: Optional[float] = None, 
                   similarity_score: Optional[float] = None, spoof_probability: Optional[float] = None,
                   deepfake_confidence: Optional[float] = None, liveness_confidence: Optional[float] = None,
                   audio_data: Optional[bytes] = None, metadata: Optional[Dict] = None):
        """
        Log an authentication attempt
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_str = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO auth_logs 
            (timestamp, user_id, result, confidence_score, similarity_score, spoof_probability,
             deepfake_confidence, liveness_confidence, audio_data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, user_id, result, confidence_score, similarity_score, spoof_probability,
              deepfake_confidence, liveness_confidence, audio_data, metadata_str))
        
        conn.commit()
        conn.close()
    
    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve authentication logs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, user_id, result, confidence_score, similarity_score, 
                   spoof_probability, deepfake_confidence, liveness_confidence, 
                   audio_data, metadata
            FROM auth_logs
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            log_entry = {
                'id': row[0],
                'timestamp': row[1],
                'user_id': row[2],
                'result': row[3],
                'confidence_score': row[4],
                'similarity_score': row[5],
                'spoof_probability': row[6],
                'deepfake_confidence': row[7],
                'liveness_confidence': row[8],
                'audio_data': row[9],
                'metadata': json.loads(row[10]) if row[10] else None
            }
            logs.append(log_entry)
        
        return logs
    
    def get_log_count(self) -> int:
        """
        Get total number of log entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM auth_logs')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def save_user_embedding(self, user_id: str, embedding: bytes):
        """
        Save user voice embedding
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE user_id = ?', (user_id,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Update existing user
            cursor.execute('''
                UPDATE users 
                SET embedding = ?, created_at = ?
                WHERE user_id = ?
            ''', (embedding, timestamp, user_id))
        else:
            # Insert new user
            cursor.execute('''
                INSERT INTO users (user_id, embedding, created_at)
                VALUES (?, ?, ?)
            ''', (user_id, embedding, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_user_embedding(self, user_id: str) -> Optional[bytes]:
        """
        Retrieve user voice embedding
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT embedding FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        return None
    
    def get_all_users(self) -> List[str]:
        """
        Get list of all registered users
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users')
        results = cursor.fetchall()
        
        conn.close()
        
        return [row[0] for row in results]
    
    def clear_logs(self):
        """
        Clear all logs (for testing purposes)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM auth_logs')
        
        conn.commit()
        conn.close()

# Global instance
log_manager = LogManager()

def log_auth_attempt(user_id: Optional[str] = None, result: Optional[str] = None, **kwargs):
    """
    Log an authentication attempt
    """
    log_manager.log_attempt(user_id=user_id, result=result, **kwargs)

def get_auth_logs(limit: int = 100):
    """
    Get authentication logs
    """
    return log_manager.get_logs(limit)

def get_log_count():
    """
    Get total number of log entries
    """
    return log_manager.get_log_count()

def save_user_embedding(user_id: str, embedding: bytes):
    """
    Save user voice embedding
    """
    log_manager.save_user_embedding(user_id, embedding)

def get_user_embedding(user_id: str) -> Optional[bytes]:
    """
    Retrieve user voice embedding
    """
    return log_manager.get_user_embedding(user_id)

def get_all_users() -> List[str]:
    """
    Get list of all registered users
    """
    return log_manager.get_all_users()

# Demo function
if __name__ == "__main__":
    # Test the log manager
    log_manager = LogManager()
    
    # Add some test logs
    log_manager.log_attempt(
        user_id="user123",
        result="ACCESS_GRANTED",
        confidence_score=0.95,
        similarity_score=0.87,
        spoof_probability=0.05,
        deepfake_confidence=0.10,
        liveness_confidence=0.90
    )
    
    log_manager.log_attempt(
        user_id="user456",
        result="SPOOF_DETECTED",
        confidence_score=0.85,
        similarity_score=0.72,
        spoof_probability=0.75,
        deepfake_confidence=0.20,
        liveness_confidence=0.85
    )
    
    # Retrieve logs
    logs = log_manager.get_logs()
    print(f"Retrieved {len(logs)} logs:")
    for log in logs:
        print(f"  {log['timestamp']}: {log['user_id']} - {log['result']} (confidence: {log['confidence_score']})")