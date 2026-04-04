from collections import deque
import time

class StressHistory:
    def __init__(self, max_window=30):
        self.window = deque(maxlen=max_window)
        self.max_window = max_window

    def add(self, score: float, confidence: float, timestamp=None):
        self.window.append({
            "score": score,
            "confidence": confidence,
            "timestamp": timestamp or time.time()
        })

    def trend(self) -> str:
        if len(self.window) < 6:
            return "Insufficient Data"
        
        half = len(self.window) // 2
        scores = [item["score"] for item in self.window]
        first_half = scores[:half]
        second_half = scores[half:]
        
        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)
        
        if mean_second - mean_first > 0.15:
            return "Rising"
        if mean_first - mean_second > 0.15:
            return "Falling"
        return "Stable"

    def sustained_level(self) -> str:
        if not self.window:
            return "Unknown"
            
        levels = []
        for item in self.window:
            score = item["score"]
            if score < 1.0:
                levels.append("Low")
            elif score < 2.0:
                levels.append("Medium")
            else:
                levels.append("High")
                
        return max(set(levels), key=levels.count)

    def average_score(self) -> float:
        if not self.window:
            return 0.0
        return sum(item["score"] for item in self.window) / len(self.window)

    def summary(self) -> dict:
        return {
            "current_score": self.window[-1]["score"] if self.window else 0.0,
            "average_score": self.average_score(),
            "trend": self.trend(),
            "sustained_level": self.sustained_level(),
            "sample_count": len(self.window),
            "window_size": self.max_window
        }

    def reset(self):
        self.window.clear()
