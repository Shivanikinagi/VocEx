import random
import string
from typing import List, Tuple

class LivenessChecker:
    """
    Liveness checker that generates random challenges and verifies spoken responses
    """
    
    def __init__(self):
        # Predefined phrases for liveness check
        self.challenge_phrases = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump",
            "Bright vixens jump dozy fowl quack",
            "Sphinx of black quartz judge my vow",
            "Watch Jeopardy Alex Trebeks fun TV quiz game",
            "Amazingly few discotheques provide jukeboxes",
            "My girl wove six dozen plaid jackets before she quit",
            "Five quacking zephyrs jolt my wax bed",
            "The five boxing wizards jump quickly"
        ]
        
        # Numbers for numeric challenges
        self.number_phrases = [
            "One two three four five",
            "Six seven eight nine zero",
            "Zero nine eight seven six five four three two one",
            "One zero zero one zero zero one",
            "Five five five eight eight eight",
            "One two three four five six seven eight nine",
            "Nine eight seven six five four three two one zero",
            "Zero one two three four five six seven eight nine"
        ]
        
        # Simple words for word challenges
        self.word_list = [
            "apple", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
            "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
            "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
            "xray", "yankee", "zulu", "alpha", "beta", "gamma", "theta", "lambda"
        ]
        
        self.current_challenge = None
        self.challenge_type = None
    
    def generate_random_phrase(self) -> str:
        """
        Generate a random phrase for liveness check
        """
        challenge_types = ["phrase", "number", "word_sequence"]
        self.challenge_type = random.choice(challenge_types)
        
        if self.challenge_type == "phrase":
            self.current_challenge = random.choice(self.challenge_phrases)
        elif self.challenge_type == "number":
            self.current_challenge = random.choice(self.number_phrases)
        else:  # word_sequence
            # Generate a sequence of 3-5 random words
            word_count = random.randint(3, 5)
            words = random.sample(self.word_list, word_count)
            self.current_challenge = " ".join(words).title()
        
        return self.current_challenge
    
    def generate_simple_challenge(self) -> str:
        """
        Generate a simple challenge (for demo purposes)
        """
        # For demo, we'll use a simple approach with random digits
        digits = [str(random.randint(0, 9)) for _ in range(4)]
        self.current_challenge = " ".join(digits)
        self.challenge_type = "digits"
        return self.current_challenge
    
    def verify_response(self, spoken_text: str, confidence_threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Verify if the spoken response matches the challenge
        Returns (is_valid, confidence_score)
        """
        if not self.current_challenge:
            return False, 0.0
        
        # Simple string matching for demo (in production, use speech-to-text comparison)
        # Normalize both texts to lowercase and remove punctuation
        import re
        
        def normalize_text(text):
            # Convert to lowercase and remove punctuation
            return re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        normalized_challenge = normalize_text(self.current_challenge)
        normalized_response = normalize_text(spoken_text)
        
        # Calculate similarity (simple approach for demo)
        if normalized_challenge == normalized_response:
            return True, 1.0
        
        # Partial matching
        challenge_words = set(normalized_challenge.split())
        response_words = set(normalized_response.split())
        
        if len(challenge_words) == 0:
            return False, 0.0
            
        # Calculate Jaccard similarity
        intersection = len(challenge_words.intersection(response_words))
        union = len(challenge_words.union(response_words))
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity >= confidence_threshold, similarity
    
    def get_current_challenge(self) -> str:
        """
        Get the current challenge phrase
        """
        return self.current_challenge or ""

# Global instance
liveness_checker = LivenessChecker()

def generate_challenge() -> str:
    """
    Generate a new liveness challenge
    """
    return liveness_checker.generate_simple_challenge()

def verify_liveness(spoken_text: str) -> Tuple[bool, float, str]:
    """
    Verify liveness of the spoken response
    Returns (is_valid, confidence_score, challenge_phrase)
    """
    is_valid, confidence = liveness_checker.verify_response(spoken_text)
    challenge = liveness_checker.get_current_challenge()
    return is_valid, confidence, challenge

# Demo function
if __name__ == "__main__":
    # Test the liveness checker
    checker = LivenessChecker()
    
    # Generate a few challenges
    for i in range(3):
        challenge = checker.generate_random_phrase()
        print(f"Challenge {i+1}: {challenge}")
        
        # Simulate verification (in real system, this would be STT output)
        is_valid, confidence = checker.verify_response(challenge)
        print(f"  Verification: {'PASS' if is_valid else 'FAIL'} (confidence: {confidence:.2f})")
        print()