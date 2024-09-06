from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from pyearth import Earth
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from pygam import LinearGAM, PoissonGAM, GammaGAM
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split

# Define a function for model selection
def model_selection_function(x, y, cross_folds, model):
    scores = []
    names = []
    # Loop through models and perform cross-validation
    for i, j in model:
        cv_scores = cross_val_score(j, x, y, cv=cross_folds, n_jobs=-1)
        scores.append(cv_scores)
        names.append(i)
    # Print the results
    for k in range(len(scores)):
        print(names[k], scores[k].mean())
    return

# Define a function for training various models
def train_models(X_train, y_train, cv, max_degree, endspan):
    try:
        # List of models to train
        models = [('lr', LinearRegression()), ('ElasticNet', ElasticNet()), ('RF', RandomForestRegressor()),
                  ('ETR', ExtraTreesRegressor()), ('GBM', GradientBoostingRegressor()), ('MLP', MLPRegressor())]

        # Perform model selection and print results
        model_selection_function(X_train, y_train, cv, models)

        print("Training MARS model")
        model = Earth(max_degree=5, endspan=20)
        
        # Cross-validate and print results for the MARS model
        scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
        print("Cross Validation score for MARS model is", scores)

        print("Training Splines model")
        spline_model = make_pipeline(SplineTransformer(n_knots=3, degree=5), LinearRegression())
        
        # Cross-validate and print results for the Splines model
        scores = cross_val_score(spline_model, X_train, y_train, cv=cv, n_jobs=-1)
        print("Cross Validation score for Splines model is", scores)

        print("Training Poisson GAM model")
        gam = PoissonGAM().gridsearch(X_train.values, y_train)
        print(gam.summary())

        print("Training Voting Regressor model")
        reg1 = LinearRegression()
        reg2 = GradientBoostingRegressor()
        reg3 = MLPRegressor()

        voting_regress = VotingRegressor(estimators=[('LR', reg1), ('GBM', reg2), ('MLP', reg3)])

        # Cross-validate and print results for the Voting Regressor model
        scores = cross_val_score(voting_regress, X_train, y_train, cv=cv)
        print("Cross Validation score for Voting Regressor model is", scores)

        print("Training Stacking Regressor model")
        reg1 = LinearRegression()
        reg2 = GradientBoostingRegressor()
        estimators = [('LR', reg1), ('GBM', reg2)]
        level_1_estimator = MLPRegressor()
        stacking_model = StackingRegressor(estimators=estimators, final_estimator=level_1_estimator)
        
        # Cross-validate and print results for the Stacking Regressor model
        scores = cross_val_score(stacking_model, X_train, y_train, cv=cv)
        print("Cross Validation score for Stacking Regressor model is", scores)

        print("Training Model Blender")
        from numpy import hstack
        X_train_1, X_val, y_train_1, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2222)

        def list_models():
            models = []
            models.append(('LR', LinearRegression()))
            models.append(('GBR', GradientBoostingRegressor()))
            return models

        def fit_all_models(models, X_train, X_val, y_train, y_val):
            level_1_feat = []
            for name, model in models:
                model.fit(X_train, y_train)
                y_hat = model.predict(X_val)
                y_hat = y_hat.reshape(len(y_hat), 1)
                level_1_feat.append(y_hat)
            level_1_feat = hstack(level_1_feat)

            level_1_estimator = MLPRegressor()
            level_1_estimator.fit(level_1_feat, y_val)
            return level_1_estimator

        def pred_data(models, blends, X_test):
            meta_model_X = []
            for name, model in models:
                yhat = model.predict(X_test)
                yhat = yhat.reshape(len(yhat), 1)
                meta_model_X.append(yhat)
            meta_model_X = hstack(meta_model_X)
            return blends.predict(meta_model_X)

        # Train a model blender and print results
        models = list_models()
        model_blender = fit_all_models(models, X_train_1, X_val, y_train_1, y_val)
        y_hat = pred_data(models, model_blender, X_train)
        
        # Cross-validate and print results for the Model Blender
        scores = cross_val_score(model_blender, X_train, y_train, cv=cv)
        print("Cross Validation score for Model Blender is", scores)

        model_gbm = GradientBoostingRegressor()
        model_gbm.fit(X_train, y_train)
    
    except Exception as e:
        print(e)

    else:
        return model_gbm, gam
