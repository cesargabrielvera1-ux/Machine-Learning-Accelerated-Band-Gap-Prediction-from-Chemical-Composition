A machine learning model that predicts experimental band gaps of inorganic materials using only chemical composition. The model achieves near-experimental accuracy (MAE = 0.424 eV) on unseen compounds through rigorous external validation.

Features

Composition-only input: Just provide the chemical formula
Fast predictions: ~1ms per compound
High accuracy: 0.424 eV MAE on external validation
Interpretable: SHAP analysis provides physical insights
Easy deployment: Minimal dependencies

#### Installation ####

# Clone the repository
git clone https://github.com/cesargabrielvera1-ux/Machine-Learning-Accelerated-Band-Gap-Prediction-from-Chemical-Composition.git
cd Machine-Learning-Accelerated-Band-Gap-Prediction-from-Chemical-Composition

# Install dependencies
pip install -r requirements.txt

# Verify installation

python -c "from bandgap_predictor import BandGapPredictor; print('Installation successful!')"







This software was created by Cesar Gabriel Vera de la Garza who is advised by Dr. Serguei Fomine.

 
