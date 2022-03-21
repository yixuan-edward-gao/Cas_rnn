# Cas_rnn
DL models for predicting editing efficiency of Cas

Original paper and raw data: https://www.science.org/doi/full/10.1126/sciadv.aax9249

Deep learning models for predicting editing efficiency of Cas proteins based on sequence data.
Processed data file not included in this repo.

## Files:
1. `model.py` implements two DL models (one RNN-based and one CNN-based)
2. `util.py` defines utility functions
3. `preprocess.py` filters and normalizes data
4. `run.py` trains and tests the model

## To run
Run `python run.py`. This trains and evaluates the model using 10-fold cross validation.
