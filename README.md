# text_nlp
Text analysis, cleaning and classification modeling
# Main Libraries used
#### Text processing
- pytesseract 0.3.9
- opencv 3.4.2
- pillow 9.0.1

#### Text classification
- pandas 1.3.4
- numpy 1.21.2
- scikit-learn 0.23.2
- xgboost 1.5.1
- nltk 3.7
- smart_open 1.8.0
- gensim 3.8.0

# Installation and setup
1. Install tesseract OCR
2. brew install tesseract (MAC)
3. sudo apt-get install tesseract-ocr (Ubuntu)
4. If you get this error: <br>
Error: The following directories are not writable by your user:
/usr/local/share/man/man8 <br>
<br>
You should change the ownership of these directories to your user.
  sudo chown -R $(whoami) /usr/local/share/man/man8
<br>
And make sure that your user has write permission.
  chmod u+w /usr/local/share/man/man8
<br>
5. Run this:
sudo chown -R $(whoami) /usr/local/share/man/man8

6. conda create -n ENV_NAME python=3.7
7. conda activate ENV_NAME
8. conda install pandas pytesseract toe cv2 pillow nltk pytest
8. conda install -c anaconda scikit-learn
<br>
For xgboost: Currently, the XGBoost package from conda-forge channel 
doesn't support GPU. There is an on-going discussion about this: 
https://github.com/conda-forge/xgboost-feedstock/issues/26. <br>
For now, you can get XGboost from one of the following here:

1. conda install -c nvidia -c rapidsai py-xgboost
2. pip install xgboost
3. conda install -c conda-forge py-xgboost-gpu
# Code Structure
model_data.py <br>
| <br>
|------> preprocess_data.py <br>
                | <br>
     data------>| <br>
# How to Run

1. Notebook: /partII/data_exploratory_analysis.ipynb
2. Preprocessing data: /partII/preprocess_data.py
3. Models: /partII/model_data.py <br>
```
python model_data.py <train_filename> 
                     <test_filename> 
                     <filedir> 
                     <steps> 
                     <text_cols>
                     <target>
```





