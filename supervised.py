import pickle

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


class supervised_models:

    def __init__(self, data, forecast_period, test_period):
        columns = data.columns.drop('Gold-T+4')

        # Select categorical columns with relatively low cardinality
        categorical_cols = [cname for cname in columns if
                            data[cname].dtype == "object"]

        # Select numerical columns
        numerical_cols = [cname for cname in columns if
                          data[cname].dtype in ['int64', 'float64']]

        X = data.drop('Gold-T+4', axis=1)
        y = data['Gold-T+4']

        n_splits = 5
        self.cv = TimeSeriesSplit(n_splits=n_splits)

        # Pre-processing for numerical data
        numerical_transformer = MinMaxScaler()

        # Pre-processing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle pre-processing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        split = test_period + forecast_period
        self.X_train = X[:-split]
        self.y_train = y[:-split]

        self.X_test = X[-split:-forecast_period]
        self.y_test = y[-split:-forecast_period]
        self.X_forecast = X[-forecast_period:]

        print("Split", split)
        print("test_period", test_period)
        print("forecast_period", forecast_period)

        print("Train data size")
        print(self.X_train.shape)

        print("Test data size")
        print(self.X_test.shape)

        print("Forecast data size")
        print(self.X_forecast.shape)


    def LinearRegression_model(self):
        model = LinearRegression()
        model_name = "Linear Regression"
        param_grid = {
            "model__fit_intercept": [True, False],
            "model__normalize": [True, False],
        }
        mae, Hyperparameters, grid = self.generic_model(model, param_grid, model_name)
        return grid, mae

    def DecisionTreeRegressor_model(self):
        model = DecisionTreeRegressor(max_depth=4,
                                      min_samples_split=5,
                                      max_leaf_nodes=10)
        model_name = "Decision Tree Regressor"
        param_grid = {
            "model__min_samples_split": [10, 20, 40],
            "model__max_depth": [2, 6, 8],
            "model__min_samples_leaf": [20, 40, 100],
            "model__max_leaf_nodes": [5, 20, 100],
        }
        mae, Hyperparameters, grid = self.generic_model(model, param_grid, model_name)
        return grid, mae

    def RandomForestRegressor_model(self):
        model = RandomForestRegressor()
        model_name = "Random Forest Regressor"
        param_grid = {"model__n_estimators": [3, 10, 30, 50],
                      "model__max_features": [2, 4, 6, 8],
                      "model__n_estimators": [3, 10],
                      "model__max_features": [2, 3, 4],
                      }
        mae, Hyperparameters, grid = self.generic_model(model, param_grid, model_name)
        return grid, mae

    def GradientBoostingRegressor_model(self):
        model = GradientBoostingRegressor(random_state=42)
        model_name = "Gradient Boosting Regressor"
        param_grid = {"model__n_estimators": [30],  # [30, 100],
                      "model__learning_rate": [0.5],
                      "model__subsample": [0.5],
                      "model__max_depth": [3],
                      }
        mae, Hyperparameters, grid = self.generic_model(model, param_grid, model_name)
        return grid, mae

    def AdaBoostRegressor_model(self):
        model = AdaBoostRegressor(random_state=42)
        model_name = "AdaBoost Regressor"
        param_grid = {"model__n_estimators": [30, 50, 100],
                      "model__learning_rate": [0.03, 0.1, 0.3],
                      }
        mae, Hyperparameters, grid = self.generic_model(model, param_grid, model_name)
        return grid, mae

    def ExtraTreesRegressor_model(self):
        model_name = "ExtraTrees Regressor"
        model = ExtraTreesRegressor(random_state=42)
        param_grid = {"model__max_depth": [30],
                      }
        mae, Hyperparameters, grid = self.generic_model(model, param_grid, model_name)
        return grid, mae

    def KNeighborsRegressor_model(self):
        model_name = "KNeighbors Regressor"
        model = KNeighborsRegressor()
        param_grid = {"model__leaf_size": [30],
                      "model__n_neighbors": [5, 10]
                      }
        mae, Hyperparameters, grid = self.generic_model(model, param_grid, model_name)
        return grid, mae

    def generic_model(self, model, param_grid, model_name):
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                   ('model', model),
                                   ])
        grid = GridSearchCV(pipeline,
                            param_grid,
                            cv=self.cv,
                            verbose=0,
                            n_jobs=-1,
                            scoring='neg_mean_absolute_error'
                            )
        print("Training", model_name)
        grid.fit(self.X_train, self.y_train)
        mae = -1 * grid.best_score_
        Hyperparameters = grid.best_params_
        best_model = grid.best_estimator_
        filename = 'models/' + model_name + '.sav'
        pickle.dump(best_model, open(filename, 'wb'))
        print("Model dumped in ", filename)
        return mae, Hyperparameters, grid

