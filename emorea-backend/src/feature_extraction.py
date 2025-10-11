'''
 # @ Author: Sofia Condesso
 # @ Create Time: 2025-03-03
 # @ Description: Feature extraction strategies.
 #
 # @ References: 
 #
 '''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import opensmile
import librosa


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 method="opensmile", 
                 sr_librosa=32000,
                 smile_feature_set="GeMAPSv01b",
                 smile_feature_level="Functionals",
                 librosa_features=("mfcc", "chroma")):
        """
        Sklearn wrapper to extract features with OpenSMILE or Librosa.
        
        Parameters
        ----------
        method : str
            'opensmile' ou 'librosa'.
        sr : int
            Sample rate para librosa.
        smile_feature_set : str
            GeMAPSv01b, eGeMAPSv02, ComParE_2016...
        smile_feature_level : str
            LLD or Functionals.
        librosa_features : tuple
            Librosa (mfcc, chroma, spectral).
        """
        self.method = method
        self.sr_librosa = sr_librosa
        self.smile_feature_set = smile_feature_set
        self.smile_feature_level = smile_feature_level
        self.librosa_features = librosa_features

        self.smile = None
        if self.method == "opensmile":
            self._init_opensmile()

    def _init_opensmile(self):
        """Initializes OpenSMILE"""
        feature_sets = {
            "GeMAPSv01b": opensmile.FeatureSet.GeMAPSv01b,
            "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
            "ComParE_2016": opensmile.FeatureSet.ComParE_2016,
        }
        feature_levels = {
            "LLD": opensmile.FeatureLevel.LowLevelDescriptors,
            "Functionals": opensmile.FeatureLevel.Functionals,
        }

        if self.smile_feature_set not in feature_sets:
            raise ValueError(f"Invalid FeatureSet: {self.smile_feature_set}")
        if self.smile_feature_level not in feature_levels:
            raise ValueError(f"Invalid FeatureLevel: {self.smile_feature_level}")

        self.smile = opensmile.Smile(
            feature_set=feature_sets[self.smile_feature_set],
            feature_level=feature_levels[self.smile_feature_level],
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.method == "opensmile":
            feats_df = self.smile.process_files(X)
            return feats_df.values

        elif self.method == "librosa":
            feats = []
            for file in X:
                y, sr = librosa.load(file, sr=self.sr_librosa)
                file_feats = []

                if "mfcc" in self.librosa_features:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    file_feats.extend(np.mean(mfcc, axis=1))

                if "chroma" in self.librosa_features:
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    file_feats.extend(np.mean(chroma, axis=1))

                if "spectral" in self.librosa_features:
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    flatness = librosa.feature.spectral_flatness(y=y)
                    file_feats.extend([
                        np.mean(centroid), np.mean(bandwidth), np.mean(flatness)
                    ])

                feats.append(file_feats)

            return np.array(feats)

        else:
            raise ValueError(f"Invalid method: {self.method}")
