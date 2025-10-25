'''
 # @ Author: Your name
 # @ Create Time: 2025-07-10 12:21:49
 # @ Modified by: Your name
 # @ Modified time: 2025-08-01 21:16:21
 # @ Description: 
 # * Fuse predictions (e.g. via weighted voting or LLM aggregation)
 # * Add a `FusionEngine` that takes all outputs and returns a final label
 '''


import numpy as np

class LateFusion:
    def __init__(self, weights=None):
        """
        weights: Dict like {'text': 0.4, 'audio': 0.3, 'video': 0.3}
        """
        self.weights = weights if weights else {'text': 1/3, 'audio': 1/3, 'image': 1/3}
        self.modalities = ['text', 'audio', 'image']
        
    def fuse(self, predictions: dict):
        """
        Find the most frequent emotion across modalities, weighted by the provided weights.
        predictions: Dict like {'text': 'happy', 'audio': 'sad', 'image': 'neutral'}

        Returns the final emotion label.
        """
        # Initialize a dictionary to hold the weighted counts
        weighted_counts = {emotion: 0 for emotion in set(predictions.values())}

        # Iterate through each modality and its prediction
        for modality, prediction in predictions.items():
            if modality in self.weights:
                weight = self.weights[modality]
                weighted_counts[prediction] += weight

        # Find the emotion with the highest weighted count
        final_emotion = max(weighted_counts, key=weighted_counts.get)
        
        return final_emotion
    
    def neutral_decision_fusion(self, results):
        """Simple rule-based fusion: override neutral with non-neutral"""
        if "text" in results and results["text"] != "neutral":
            return results["text"]
        if "speech" in results and results["speech"] != "neutral":
            return results["speech"]
        if "face" in results and results["face"].get("dominant_emotion") != "neutral":
            return results["face"]["dominant_emotion"]
        return "neutral"
        
