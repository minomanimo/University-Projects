from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

def build_linear_pipeline(categorical_columns: list[str], numeric_columns: list[str], config: dict) -> Pipeline:
    # TODO:
    # Build a preprocessing pipeline for the linear baseline.
    # Your pipeline should handle:
    # - categorical features
    # - numerical features
    #
    categorical_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore'))      
    ])
    numeric_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())        
    ])
    # Then create a Logistic Regression model and return a single sklearn Pipeline.
    #
    # Suggested sklearn components:
    # - ColumnTransformer
    # - SimpleImputer
    # - OneHotEncoder
    # - StandardScaler
    # - LogisticRegression
    preprocessor = ColumnTransformer([
        ('categorical', categorical_pipeline, categorical_columns),
        ('numeric', numeric_pipeline, numeric_columns)
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=config['C'],max_iter=config['max_iter'], class_weight=config['class_weight']))
    ])
    return model
