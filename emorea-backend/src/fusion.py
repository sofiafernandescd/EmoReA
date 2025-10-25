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
from collections import defaultdict, Counter

class LateFusion:
    def __init__(self, 
                 weights={'text': 0.4, 'audio': 0.3, 'video': 0.3}, 
                 neutral_label="neutral"
        ):
        """
        weights: dict (e.g., {'text': 0.4, 'audio': 0.3, 'video': 0.3})
        neutral_label: label used for neutral emotion (default='neutral')
        """
        self.weights = weights or {'text': 1/3, 'audio': 1/3, 'video': 1/3}
        self.modalities = list(self.weights.keys())
        self.neutral_label = neutral_label

    def _normalize_confidences(self, confidences):
        """Normalize probabilities to sum to 1."""
        total = sum(confidences.values())
        if total == 0:
            return {k: 1/len(confidences) for k in confidences}
        return {k: v / total for k, v in confidences.items()}

    def _aggregate_video(self, video_outputs):
        """
        Aggregate multiple frame-level predictions (labels or probability dicts)
        into a single averaged emotion distribution.
        """
        if not video_outputs:
            return {self.neutral_label: 1.0}

        # if is list of dicts (probabilities)
        if isinstance(video_outputs[0], dict):
            agg = defaultdict(float)
            for frame_pred in video_outputs:
                normed = self._normalize_confidences(frame_pred)
                for emotion, prob in normed.items():
                    agg[emotion] += prob
            return self._normalize_confidences({k: v / len(video_outputs) for k, v in agg.items()})

        # if is list of labels
        else:
            counts = Counter(video_outputs)
            return self._normalize_confidences({k: v / len(video_outputs) for k, v in counts.items()})


    def fuse(self, predictions, top_k=3):
        """
        Fuse predictions from multiple modalities.
        
        predictions: dict
            {
              'text': {'happy': 0.8, 'sad': 0.1, 'neutral': 0.1},
              'audio': 'sad',
              'video': [ {'happy':0.4,'neutral':0.6}, {'happy':0.3,'neutral':0.7} ]
            }

        Returns:
            final_label (str)
            fused_probs (dict)
            top_k_list (list of tuples)
        """
        combined = defaultdict(float)

        for modality, output in predictions.items():
            weight = self.weights.get(modality, 0)

            # modality output type
            if isinstance(output, list):  # frame-wise
                confs = self._aggregate_video(output)
            elif isinstance(output, dict):  # probability distribution
                confs = self._normalize_confidences(output)
            elif isinstance(output, str):  # single label
                confs = {output: 1.0}
            else:
                raise ValueError(f"Unsupported output type for modality {modality}: {type(output)}")

            # weighted accumulation
            for emotion, prob in confs.items():
                combined[emotion] += weight * prob

        # normalize final fused probabilities
        fused_probs = self._normalize_confidences(combined)
        final_label = max(fused_probs, key=fused_probs.get)

        # compute top-k
        sorted_emotions = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_list = sorted_emotions[:top_k]

        return final_label, fused_probs, top_k_list

    # optional neutral rule
    def rule_based_neutral_override(self, predictions, fused_label):
        """
        Optional rule: if all modalities predict 'neutral', return neutral;
        otherwise keep the fused label.
        """
        non_neutrals = [
            p for p in predictions.values()
            if (isinstance(p, str) and p != self.neutral_label)
            or (isinstance(p, dict) and max(p, key=p.get) != self.neutral_label)
        ]

        if not non_neutrals:
            return self.neutral_label
        return fused_label