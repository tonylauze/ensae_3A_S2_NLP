# Sentiment Analysis on IMDb Movie Reviews

This project was completed as part of the NLP course in the 3rd year at ENSAE, supervised by Christopher KERMORVANT.

`Report_Tony_Lauze` contains the pdf version of the report associated with the project.

Based on the paper *Learning Word Vectors for Sentiment Analysis* by Maas et al. (2011), this project aims to compare classic machine learning algorithms (logistic regression and SVM applied to text) with more recent LLM models for binary sentiment classification (positive/negative) of movie review comments.

The database used is the one introduced by Maas et al. (2011), consisting of 50,000 movie reviews equally distributed between positive and negative opinions, obtained from the IMDb website.

## Repository Structure

The `paper` folder contains the article by Maas et al. (2011).

The `output` folder contains the outputs presented in the report.

The `models` folder contains the classification models trained in notebook 2_prediction_benchmark_LSA.

A `data` folder is created by running notebook `0_import_data.ipynb` but is ignored by gitignore.

## Detailed Description of Notebooks

### `0_import_data.ipynb`
Import and preparation of the IMDb dataset introduced by Maas et al. (2011). This notebook:
- Loads raw movie review data
- Creates two dataframes (df_train and df_test) used in the following notebooks
- Saves these dataframes in parquet format in the `/data` folder

### `1_descriptive_analysis_LDA.ipynb`
Descriptive analysis of the dataset using the unsupervised technique of Latent Dirichlet Allocation (LDA) to discover latent themes in the reviews. This notebook:
- Applies LDA to extract 10 main topics
- Visualizes the most frequent words per topic using word clouds
- Suggests the limitations of LDA for capturing sentiment polarity
The word clouds and descriptive graphs are saved in the output folder.

### `2_prediction_benchmark_LSA.ipynb`
This notebook uses a method inspired by the paper by Maas et al. (2011) to establish benchmark results in sentiment classification. The classification follows these steps:
- Converting texts into numerical vectors with TF-IDF
- Dimensionality reduction through LSA (Latent Semantic Analysis)
- Training via cross-validation, and evaluation of two classification models:
  - Naive Bayes (MultinomialNB)
  - Support Vector Machine (SVC)
- The confusion matrices are saved in the output folder

### `3_prediction_BERT.ipynb`
In this notebook, we use a DistilBERT model that is specifically fine-tuned for binary sentiment classification on the SST-2 database. This is the "distilbert-base-uncased-finetuned-sst-2-english" model loaded from Hugging Face.
This model is applied directly to our dataset and the confusion matrix is saved in the output folder.

## Installation and Execution

- Clone the code: `git clone https://github.com/tonylauze/ensae_3A_S2_NLP.git`  
- Create a virtual environment: `virtualenv -p python3 nlp-env`  
- Activate the virtual environment: `source nlp-env/bin/activate`  
- Install the required packages: `pip install -r requirements.txt`  
- Register the virtualenv with jupyter: `python -m ipykernel install --name=nlp-env`  
- Change the kernel to nlp-env  
- Run the notebooks in order (0, 1, 2, 3) to reproduce the complete analysis.

**Important note**: All notebooks must be executed in the Python virtual environment that contains the dependencies specified in `requirements.txt`. The trained models (SVC and logistic regression) have been saved in the `models` folder and can be loaded directly into the notebooks without retraining.
