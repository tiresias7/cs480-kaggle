import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from autogluon.tabular import TabularPredictor

SEED = 42
TARGET_COLUMNS = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']

# Set seed for reproducibility
np.random.seed(SEED)

# Load the data
train_combined_df = pd.read_csv('./dinov2/train_features_combined.csv')
val_combined_df = pd.read_csv('./dinov2/val_features_combined.csv')
test_combined_df = pd.read_csv('./dinov2/test_features_combined.csv')
train_df = pd.read_csv('./dinov2/train_splitted.csv')
val_df = pd.read_csv('./dinov2/val_splitted.csv')
test_df = pd.read_csv('./dinov2/test_splitted.csv')

# Initialize a dictionary to store models and scores for each trait
models = {}
scores = {}

# Train and evaluate a model for each trait using AutoGluon
for trait in TARGET_COLUMNS:
    print(f"Training model for {trait} using AutoGluon...")
    
    # Initialize and train the AutoGluon predictor
    train_combined_df_with_trait = train_combined_df.copy()
    train_combined_df_with_trait[trait] = train_df[trait]
    predictor = TabularPredictor(
        label=trait, 
        problem_type='regression', 
        eval_metric='r2', 
        path=f"./DinoV2-with-AutoTabular-{trait}",
    ).fit(train_data=train_combined_df_with_trait)
    
    # Store the trained model
    models[trait] = predictor
    
    # Make predictions on the validation set
    val_predictions = predictor.predict(val_combined_df)
    
    # Evaluate the model
    score = r2_score(val_df[trait], val_predictions)
    scores[trait] = score
    print(f"R2 score for {trait}: {score:.4f}")

print(f"Mean R2 score: {np.mean(list(scores.values())):.4f}")


submission = pd.DataFrame({'id': test_df['id']})
submission[TARGET_COLUMNS] = 0
for trait in TARGET_COLUMNS:
    submission[trait] = models[trait].predict(test_combined_df)

submission.columns = submission.columns.str.replace('_mean', '')
submission.to_csv('submission-final.csv', index=False)
print("Submission file created.")
