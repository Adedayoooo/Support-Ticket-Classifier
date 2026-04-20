from pathlib import Path

ROOT_DIR=Path(__file__).parent.parent

#Main paths
APP_DIR=ROOT_DIR/"app"
DATA_DIR=ROOT_DIR/"data"
SRC_DIR=ROOT_DIR/"src"
MODEL_DIR=ROOT_DIR/"model"
REQUIREMENTS_PATH=ROOT_DIR/"requirements.txt"
HF_TOKEN_PATH=ROOT_DIR/"hf_token.txt"

#Data files
TRAIN_DATA_PATH=DATA_DIR/"train.csv"
PROCESSED_DATA=DATA_DIR/"data.joblib"

#App file
APP_PATH=APP_DIR/"app.py"

#Model file
MODEL_PATH=MODEL_DIR/"model.safetensors"

#Training code files
TRAIN_PATH=SRC_DIR/"train.py"

#Test code files
CLASSIFSY_PATH=SRC_DIR/"classify.py"

# Config files
CONFIG_PATH=SRC_DIR/"config.json"

#Preprocessing file
PREPROCESSING_PATH=SRC_DIR/"preprocessing.py"