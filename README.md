# What is it?

A machine learning model that predicts experimental band gaps of inorganic materials using only chemical composition. The model achieves near-experimental accuracy (MAE = 0.424 eV) on unseen compounds through rigorous external validation.

Features:

Composition-only input: Just provide the chemical formula

Fast predictions: ~1ms per compound

High accuracy: 0.424 eV MAE on external validation

Easy deployment: Minimal dependencies

Limitaitons:

Polymorph Problem: Cannot distinguish between different crystal structures of the same composition

Band Gap Range: Accuracy decreases for wide-bandgap materials (>4 eV)

Composition Only: Does not consider structural information or synthesis conditions

Inorganic Focus: Optimized for inorganic crystalline materials

# Quick start

Assuming you are using Google Colab...

1.- Make sure the model "xgboost_final_model.joblib" is available in your session tipically /content/ directory.

2.- Please run the following before using the #Script:

!pip install pymatgen

pymatgen is MANDATORY for our script to run 

3.- In the provided script (see below) there is a section marked with several #### symbols.
The Section is called: ########### Compositions to predict ##########
There you can modify the provided examples with chemical compositions you want to explore between " ".

Feel free to copy and paste the following script and then running it!

# Script:

#predict_bandgap.py



import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition # Changed import statement
from matminer.featurizers.composition import ElementProperty

class BandGapPredictor:
    """
    Predict experimental band gaps from chemical composition using XGBoost.
    """

    def __init__(self, model_path='xgboost_final_model.joblib'):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained XGBoost model (.joblib file)
        """
        self.model = joblib.load(model_path)
        self.featurizer = ElementProperty.from_preset("magpie")
        self.feature_names = None

    def predict(self, composition_list):
        """
        Predict band gaps for a list of chemical compositions.

        Args:
            composition_list: List of chemical formulas (e.g., ["TiO2", "GaAs"])

        Returns:
            predictions: Array of predicted band gaps in eV
        """
        # Featurize compositions
        features = self._featurize_compositions(composition_list)

        # Align features with training data
        features_aligned = self._align_features(features)

        # Make predictions
        predictions = self.model.predict(features_aligned)

        return predictions

    def predict_single(self, composition):
        """
        Predict band gap for a single chemical composition.

        Args:
            composition: Chemical formula (e.g., "TiO2")

        Returns:
            band_gap: Predicted band gap in eV
        """
        return self.predict([composition])[0]

    def _featurize_compositions(self, composition_list):
        """
        Convert chemical formulas to Magpie features.
        """
        features_list = []

        for comp_str in composition_list:
            # Convert to pymatgen Composition
            comp = Composition(comp_str)

            # Generate Magpie features
            features = self.featurizer.featurize(comp)
            features_list.append(features)

        # Convert to DataFrame
        feature_names = self.featurizer.feature_labels()
        df = pd.DataFrame(features_list, columns=feature_names)

        # Store feature names (first time only)
        if self.feature_names is None:
            self.feature_names = feature_names

        return df

    def _align_features(self, features_df):
        """
        Ensure features match the training data format.
        """
        # Fill any missing values
        features_df = features_df.fillna(0)

        # Ensure all columns are numeric
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        features_df = features_df.fillna(0)

        return features_df


def main():
    """
    Example usage of the BandGapPredictor.
    """
    # Initialize predictor
    predictor = BandGapPredictor('xgboost_final_model.joblib')

    ########### Compositions to predict ##########
    test_compositions = [
        "TiO2",      # Titanium dioxide
        "GaAs",      # Gallium arsenide
        "Si",        # Silicon
        "ZnO",       # Zinc oxide
        "CdTe",      # Cadmium telluride
        "CH3NH3PbI3" # Perovskite (MAPbI3)
    ]

    # Predict band gaps
    print("🚀 Predicting Band Gaps...")
    print("-" * 50)

    predictions = predictor.predict(test_compositions)

    for comp, pred in zip(test_compositions, predictions):
        print(f"{comp:15} → {pred:.3f} eV")

    print("-" * 50)
    print(f"✅ Predictions complete!")

    return predictions


if __name__ == "__main__":
    main()


# Important

This software was created by Cesar Gabriel Vera de la Garza who is advised by Dr. Serguei Fomine.


If you use this code or model in your research, please cite:

@article{veradelagarza2025,
  title={Machine-Learning-Accelerated Band Gap Prediction from Chemical Composition with Near-Experimental Accuracy},
  author={Vera de la Garza, Cesar Gabriel and Fomine, Serguei},
  journal={Next Materials},
  year={2025},
  doi={TBD}
}

For Methodology:

@article{matbench2020,
  title={Matbench: A benchmark for materials property prediction},
  author={Dunn, Alexander and Wang, Qi and Ganose, Alex and Dopp, Daniel and Jain, Anubhav},
  journal={npj Computational Materials},
  volume={6},
  number={1},
  pages={138},
  year={2020}
}

@article{magpie2016,
  title={The Materials Agnostic Platform for Informatics and Exploration (Magpie)},
  author={Ward, Logan and Agrawal, Ankit and Choudhary, Alok and Wolverton, Christopher},
  journal={Computational Materials Science},
  volume={125},
  pages={145--150},
  year={2016}
}
