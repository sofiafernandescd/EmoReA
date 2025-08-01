import numpy as np

class LateFusion:
    def __init__(self, weights=None):
        """
        weights: Dict like {'text': 0.4, 'audio': 0.3, 'video': 0.3}
        """
        self.weights = weights if weights else {'text': 1/3, 'audio': 1/3, 'video': 1/3}
        self.modalities = ['text', 'audio', 'video']
        
    def fuse(self, predictions: dict):
        """
        predictions: Dict of modality -> softmax vector (np.array)
        Returns: fused_prediction (np.array), confidence_per_modality (dict)
        """
        assert set(predictions.keys()) == set(self.modalities), "All modalities must be present"
        
        fused = np.zeros_like(predictions['text'])
        confidences = {}
        
        for modality in self.modalities:
            conf_score = np.max(predictions[modality])
            confidences[modality] = conf_score
            fused += self.weights[modality] * predictions[modality]
        
        final_pred = np.argmax(fused)
        return final_pred, fused, confidences
