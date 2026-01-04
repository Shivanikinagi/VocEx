import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class AlertSystem:
    """
    Emergency alert system for security events
    """
    
    def __init__(self, config_file: str = "alert_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.alert_count = 0
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load alert configuration from file
        """
        default_config = {
            "email_enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "",
            "sender_password": "",
            "recipient_emails": [],
            "alert_threshold": 3,  # Number of alerts before sending email
            "dashboard_alerts": True
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with default config
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def save_config(self):
        """
        Save current configuration to file
        """
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def send_email_alert(self, alert_type: str, details: Dict[str, Any], user_id: str = None):
        """
        Send email alert for security events
        """
        if not self.config.get("email_enabled", False):
            return False
        
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"SECURITY ALERT: {alert_type}"
            message["From"] = self.config["sender_email"]
            message["To"] = ", ".join(self.config["recipient_emails"])
            
            # Create text content
            text_content = f"""
SECURITY ALERT DETECTED

Type: {alert_type}
User ID: {user_id or 'Unknown'}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Details: {json.dumps(details, indent=2)}

This is an automated security alert from VoiceAuth Pro system.
Please investigate this incident immediately.
            """
            
            # Create HTML content
            html_content = f"""
            <html>
              <body>
                <h2>SECURITY ALERT DETECTED</h2>
                <p><strong>Type:</strong> {alert_type}</p>
                <p><strong>User ID:</strong> {user_id or 'Unknown'}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Details:</strong></p>
                <pre>{json.dumps(details, indent=2)}</pre>
                <p><em>This is an automated security alert from VoiceAuth Pro system.<br>
                Please investigate this incident immediately.</em></p>
              </body>
            </html>
            """
            
            # Turn these into plain/html MIMEText objects
            part1 = MIMEText(text_content, "plain")
            part2 = MIMEText(html_content, "html")
            
            # Add HTML/plain-text parts to MIMEMultipart message
            message.attach(part1)
            message.attach(part2)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"]) as server:
                server.starttls(context=context)
                server.login(self.config["sender_email"], self.config["sender_password"])
                server.sendmail(
                    self.config["sender_email"], 
                    self.config["recipient_emails"], 
                    message.as_string()
                )
            
            print(f"Email alert sent for {alert_type}")
            return True
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
            return False
    
    def trigger_alert(self, alert_type: str, details: Dict[str, Any], user_id: str = None):
        """
        Trigger security alert
        """
        self.alert_count += 1
        
        # Log the alert
        alert_log = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "user_id": user_id,
            "details": details,
            "alert_id": self.alert_count
        }
        
        # Save alert to log file
        log_file = "security_alerts.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(alert_log) + '\n')
        
        print(f"SECURITY ALERT: {alert_type} for user {user_id}")
        print(f"Details: {details}")
        
        # Send email if threshold reached
        if (self.config.get("email_enabled", False) and 
            self.alert_count % self.config.get("alert_threshold", 3) == 0):
            self.send_email_alert(alert_type, details, user_id)
        
        return alert_log
    
    def get_recent_alerts(self, limit: int = 50) -> list:
        """
        Get recent security alerts
        """
        alerts = []
        log_file = "security_alerts.log"
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Get last N lines
                    recent_lines = lines[-limit:] if len(lines) > limit else lines
                    for line in recent_lines:
                        alerts.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Error reading alert log: {e}")
        
        return alerts[::-1]  # Return in reverse chronological order
    
    def get_alert_count(self) -> int:
        """
        Get total number of alerts
        """
        return self.alert_count
    
    def reset_alert_count(self):
        """
        Reset alert counter
        """
        self.alert_count = 0

# Global instance
alert_system = AlertSystem()

def trigger_security_alert(alert_type: str, details: Dict[str, Any], user_id: str = None):
    """
    Trigger a security alert
    """
    return alert_system.trigger_alert(alert_type, details, user_id)

def get_recent_security_alerts(limit: int = 50):
    """
    Get recent security alerts
    """
    return alert_system.get_recent_alerts(limit)

def get_alert_count():
    """
    Get total number of alerts
    """
    return alert_system.get_alert_count()

# Demo function
if __name__ == "__main__":
    # Test the alert system
    alert_system = AlertSystem()
    
    # Trigger some test alerts
    alert_system.trigger_alert(
        "SPOOF_DETECTED",
        {"confidence": 0.85, "spoof_probability": 0.75},
        "test_user"
    )
    
    alert_system.trigger_alert(
        "DEEPFAKE_DETECTED",
        {"confidence": 0.92},
        "test_user2"
    )
    
    # Retrieve recent alerts
    alerts = alert_system.get_recent_alerts()
    print(f"Retrieved {len(alerts)} recent alerts:")
    for alert in alerts[:3]:  # Show first 3
        print(f"  {alert['timestamp']}: {alert['type']} for {alert['user_id']}")