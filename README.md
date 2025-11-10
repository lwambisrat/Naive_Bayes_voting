# Naive Bayes Classifier for Congressional Voting Records

This project implements a **Naive Bayes classifier from scratch** to classify members of the U.S. Congress as **Democrats or Republicans** based on their voting records. The dataset contains 16 categorical attributes representing votes on key issues. The implementation handles **missing values** and uses **10-fold cross-validation** to evaluate model performance.

---

## Tables of content

- Features
- Requirements & setup instructions
- Project structure
- Usage

---

## Features
- Naive Bayes classifier implemented **without external ML libraries** (pure Python).  
- Handles **categorical and missing values** using class-wise mode replacement.  
- Performs **10-fold cross-validation** and computes the mean accuracy.  
- Prints per-fold accuracy and overall average accuracy.  
- Modular and well-commented code for easy understanding and modification.

---

## Requirements & setup instructions
- Python 3.8 or higher  
- Dataset: [Congressional Voting Records – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/105/congressional+voting+records)
- Clone the repository
  
   Using SSH:

   ` git clone git@github.com:lwambisrat/Naive_Bayes_voting.git`
   ` cd Naive_Bayes_voting`

   Or using HTTPS:
  
    ` git clone https://github.com/lwambisrat/ID3_BreastCancer.git`
    ` cd ID3_BreastCancer `
- Standard libraries only: `csv`, `random`, `collections` (`Counter`, `defaultdict`)  


---

## Project Structure

Naive_Bayes_voting/

├── data/

    └── house-votes-84.data # Voting dataset
    └── house-votes-84.names # Dataset description
    

├── naive_bayes_voting.py

├── README.md 



---

## Usage

Run the Naive Bayes classifier and evaluate with 10-fold cross-validation:

     `python naive_bayes.py`

The script will output:

- Number of records loaded

- Missing value handling summary

- Accuracy for each fold

- Mean accuracy across all folds
  
- The out put is going to be same format with below logs

      Loaded 435 records from dataset.
      Missing values handled using class-wise mode replacement.

       Fold 1 Accuracy: 0.909
       Fold 2 Accuracy: 0.932
       Fold 3 Accuracy: 0.932
       Fold 4 Accuracy: 0.864
       Fold 5 Accuracy: 0.864
       Fold 6 Accuracy: 0.930
       Fold 7 Accuracy: 0.953
       Fold 8 Accuracy: 0.907
       Fold 9 Accuracy: 0.884
       Fold 10 Accuracy: 0.977

      Average Accuracy over 10 folds: 0.915
