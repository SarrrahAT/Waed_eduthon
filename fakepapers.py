import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Constants
PATH_TO_DATA = 'C:/Users/HP/Desktop/Eduthon/'
INDEX_COL_NAME = 'id'
INPUT_COL_NAME = 'text'
TARGET_COL_NAME = 'fake'

# Reading data
train_df = pd.read_csv(PATH_TO_DATA + "fake_papers_train_part_public.csv", index_col=INDEX_COL_NAME)
test_df = pd.read_csv(PATH_TO_DATA + "fake_papers_test_public.csv", index_col=INDEX_COL_NAME)

# Defining the model
tfidf_transformer = TfidfVectorizer(
    ngram_range=(1, 2),
    analyzer='word',
    lowercase=True,
    max_features=50000,
    stop_words='english'
)

logreg = LogisticRegression(
    C=1,
    random_state=17,
    solver='lbfgs',
    n_jobs=4,
    max_iter=500
)

model = Pipeline([
    ('tfidf', tfidf_transformer), 
    ('logit', logreg)
])

# Defining the validation scheme
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

# Running the model
cv_f1_scores = []
skf_split_generator = skf.split(X=train_df[INPUT_COL_NAME], y=train_df[TARGET_COL_NAME])

for fold_id, (train_idx, val_idx) in enumerate(skf_split_generator, 1):
    curr_train_df = train_df.iloc[train_idx]
    curr_val_df = train_df.iloc[val_idx]
    
    model.fit(X=curr_train_df[INPUT_COL_NAME], y=curr_train_df[TARGET_COL_NAME])
    
    # Making predictions for the current validation set
    curr_preds = model.predict(X=curr_val_df[INPUT_COL_NAME])
    curr_f1 = f1_score(y_true=curr_val_df[TARGET_COL_NAME], y_pred=curr_preds)
    print(f"F1-score for fold {fold_id}: {curr_f1:.3f}")
    cv_f1_scores.append(curr_f1)

# Average cross-validation F1-score
print(f'Average cross-validation F1-score: {np.mean(cv_f1_scores):.3f} +/- {np.std(cv_f1_scores):.3f}')

# Final training on the entire training set
model.fit(X=train_df[INPUT_COL_NAME], y=train_df[TARGET_COL_NAME])

# Making predictions for the test set
test_preds = model.predict(test_df[INPUT_COL_NAME])

# Output predictions (adjust as per your submission format)
output_df = pd.DataFrame({'id': test_df.index, 'fake': test_preds})
output_df.to_csv('submission.csv', index=False)

# Example of using the model for user input (optional)
def check_user_input(user_input):
    prediction = model.predict([user_input])[0]
    return "FAKE" if prediction == 1 else "NOT FAKE"

# Example usage
user_input_text = input("Enter the text to be checked: ")
prediction = check_user_input(user_input_text)
print(f"The input text is: {prediction}")
