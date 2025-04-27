
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ----------------------- Helper Functions -----------------------

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def impute_missing(df, feature_handling):
    for feature, details in feature_handling.items():
        if details['is_selected']:
            if details['feature_details']['missing_values'] == 'Impute':
                method = details['feature_details']['impute_with']
                value = details['feature_details']['impute_value']
                if method == 'Average of values':
                    imputer = SimpleImputer(strategy='mean')
                    df[[feature]] = imputer.fit_transform(df[[feature]])
                elif method == 'custom':
                    df[feature] = df[feature].fillna(value)
    return df

def generate_features(df, feature_generation):
    if 'linear_interactions' in feature_generation:
        for pair in feature_generation['linear_interactions']:
            new_col = f'{pair[0]}_plus_{pair[1]}'
            df[new_col] = df[pair[0]] + df[pair[1]]
    if 'polynomial_interactions' in feature_generation:
        for formula in feature_generation['polynomial_interactions']:
            f1, f2 = formula.split('/')
            new_col = f'{f1}_div_{f2}'
            df[new_col] = df[f1] / (df[f2].replace(0, np.nan))
    if 'explicit_pairwise_interactions' in feature_generation:
        for formula in feature_generation['explicit_pairwise_interactions']:
            f1, f2 = formula.split('/')
            new_col = f'{f1}_times_{f2}'
            df[new_col] = df[f1] * df[f2]
    return df

def reduce_features(method, X, y, json_params):
    if method == 'No Reduction':
        return X
    elif method == 'Corr with Target':
        correlations = X.apply(lambda col: abs(col.corr(y)))
        top_features = correlations.sort_values(ascending=False).head(int(json_params['num_of_features_to_keep'])).index.tolist()
        return X[top_features]
    elif method == 'Tree-based':
        model = RandomForestRegressor(n_estimators=int(json_params['num_of_trees']), max_depth=int(json_params['depth_of_trees']), random_state=42)
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(int(json_params['num_of_features_to_keep'])).index.tolist()
        return X[top_features]
    elif method == 'PCA':
        pca = PCA(n_components=int(json_params['num_of_features_to_keep']))
        X_pca = pca.fit_transform(X)
        return pd.DataFrame(X_pca)
    else:
        raise ValueError('Unknown feature reduction method')

def select_models(algorithms, prediction_type):
    selected_models = []
    params_grid = []
    if prediction_type.lower() == 'regression':
        for algo_name, algo_params in algorithms.items():
            if algo_params['is_selected']:
                if algo_name == 'RandomForestRegressor':
                    model = RandomForestRegressor()
                    param_grid = {
                        'n_estimators': list(range(algo_params['min_trees'], algo_params['max_trees']+1, 5)),
                        'max_depth': list(range(algo_params['min_depth'], algo_params['max_depth']+1, 1)),
                        'min_samples_leaf': list(range(algo_params['min_samples_per_leaf_min_value'], algo_params['min_samples_per_leaf_max_value']+1, 1))
                    }
                    selected_models.append(model)
                    params_grid.append(param_grid)
    return selected_models, params_grid

def evaluate_model(y_true, y_pred):
    print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

# ----------------------- Main Execution -----------------------

def main():
    # Load JSON configuration
    json_data = load_json('algoparams_from_ui.json')
    config = json_data['design_state_data']

    # Load Dataset
    df = pd.read_csv('iris.csv')

    # Feature Handling (Imputation)
    df = impute_missing(df, config['feature_handling'])

    # Feature Generation
    df = generate_features(df, config['feature_generation'])

    # Define Target and Features
    target_col = config['target']['target']
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Feature Reduction
    feature_reduction = config['feature_reduction']['feature_reduction_method']
    X = reduce_features(feature_reduction, X, y, config['feature_reduction'])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    models, grids = select_models(config['algorithms'], config['target']['prediction_type'])

    # Iterate over each model
    for model, param_grid in zip(models, grids):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        grid = GridSearchCV(pipe, param_grid={'model__' + k: v for k, v in param_grid.items()}, 
                            cv=5, n_jobs=-1, scoring='r2')
        grid.fit(X_train, y_train)

        print(f"Best Params for {model.__class__.__name__}: {grid.best_params_}")

        y_pred = grid.predict(X_test)

        evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
