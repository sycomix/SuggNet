# %%
# WIP: feature importance analysis
# This is just a proof-of-concept. do not even read the code :~)
# Use VSCode or Jupytext to run the notebook

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, TweedieRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
import shap
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# relative path to the project root directory
project_dir = Path('.')

# the following line ensures we have access to the behaverse analysis package
# from python.behaverse.datasets import load_bids_dataset
# from python.behaverse.preprocessing.utils import squeeze_listlike_columns

# ------------------------


class RandomHiddenLayerSizes(scipy.stats.rv_discrete):
  """Generates randomized MLP layer sizes; within each layer exists random number of nodes."""

  def __init__(self,
               min_layers=1,
               max_layers=4,
               min_nodes_per_layer=10,
               max_nodes_per_layer=100,
               *args, **kwargs):

    super().__init__(*args, **kwargs)
    self.min_layers = max(1, min_layers)
    self.max_layers = max(1, max_layers)
    self.min_nodes_per_layer = max(1, min_nodes_per_layer)
    self.max_nodes_per_layer = max(1, max_nodes_per_layer)

  def rvs(self, *args, **kwargs):
    return tuple(
        np.random.randint(
            self.min_nodes_per_layer,
            self.max_nodes_per_layer,
            size=np.random.randint(self.min_layers, self.max_layers),
        ))


# ------------------------
# let's define a function that takes care of preprocessing, modeling, tuning, and resampling.
# ------------------------

def fit_and_tune_model(X, y, CV_SPLITS=5, scoring=None):
  """A generic integrated pipeline that handles preprocessing, modeling, tuning, and resampling.

  Args:
      X (DataFrame): predictors.
      y (DataFrame): outcomes.
      CV_SPLITS (int, optional): Number of cross-validation folds. It will be use for both test/train
        splitting and analysis/assessment folds. Defaults to 5.

  Returns:
      (model, X_train, X_test, y_train, y_test): A tuple of the best tuned model and test/train splits.
  """

  # split the data into train and test. Test set won't be used.
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/CV_SPLITS, random_state=0)
  X_train, X_test, y_train, y_test  = X,X,y,y
 

  # categorize features into labels, categorical features, and numerical features.
  lbl_features = [c for c in X.columns if c.endswith('_index')]
  cat_features = X.select_dtypes(['object']).columns.to_list()
  num_features = X.select_dtypes(['float64', 'int64']).columns.to_list()
  num_features = list(set(num_features) - set(lbl_features))

  # pipeline to preprocess label features
  lbl_pipe = Pipeline([
      ('lbl', OrdinalEncoder())
  ])

  # pipeline to preprocess categorical features
  cat_pipe = Pipeline([
      ('ohe', OneHotEncoder(handle_unknown='ignore'))
  ])

  # pipeline to preprocess numerical features
  num_pipe = Pipeline([
      ('imputer', SimpleImputer(strategy='mean'))
  ])

  # preprocessing pipeline (combines pipelines for labels, categories, and numerical features)
  prep_pipe = ColumnTransformer([
      # ('lbl', lbl_pipe, lbl_features),
      ('cat', cat_pipe, cat_features),
      ('num', num_pipe, num_features)
  ])

  # main modeling pipeline (values are temporary; will be redefined by the tuning part)
  pipeline = Pipeline([
      ('preprocess', prep_pipe),
      ('scaler', StandardScaler()),
      # ('reduce_dim', PCA(20)),
      ('estimator', SVR(kernel="rbf")),
      # PLSRegression()),
  ])

  # define a space of all possible model parameters to search for using grid and randomized tuning.
  # Some of the models take time to be fitted; comment them if you're debugging.
  params_space = [{
  #     # standard models (scale -> reduce dimensionality -> estimate multiple outcomes)
  #     'scaler': [None],  # , RobustScaler(), QuantileTransformer()],
  #     'reduce_dim': [None], #[PCA(n_components=n) for n in np.arange(1, len(X.columns) - 1)],
  #     'estimator': [TweedieRegressor(power=0, link='auto', fit_intercept=p) for p in [True, False]]
  # },
  # {
  #     # PLS model (scale -> estimate outcomes)
  #     'scaler': [None, StandardScaler(), RobustScaler()],
  #     'reduce_dim': [None],
  #     'estimator': [PLSRegression(n_components=p) for p in np.arange(1, len(X.columns))]
  # },
  # {
#       # Multilayer neural network model (scale -> estimate using randomized hidden layers)
#       'scaler': [StandardScaler()],
#       'reduce_dim': [None],
#       'estimator': [MLPClassifier()],
#       # 'estimator__hidden_layer_sizes': [RandomHiddenLayerSizes()],
#       # scipy random distribution
#       # 'estimator__hidden_layer_sizes': scipy.stats.poisson(10, 100),
#       # numpy randomizer (grid search)
#       'estimator__hidden_layer_sizes': [np.random.randint(10, 100, size=n_layers) for n_layers in [1, 2, 3]]
#   },
#   {
#     # Multilayer neural network model (scale -> estimate using randomized hidden layers)
#     'scaler': [StandardScaler()],
#     'reduce_dim': [None],
#     'estimator': [RandomForestClassifier()],
  
# },
# {
    # Multilayer neural network model (scale -> estimate using randomized hidden layers)
    'scaler': [StandardScaler()],
    'reduce_dim': [None],
    'estimator': [SVR(kernel="linear")]
  
    }]

  # define the parameter tuning search and fit it (uses -MSE to score and compare models)
  # grid_search = GridSearchCV(pipeline,
  #                                  params_space,
  #                                  cv=CV_SPLITS,
  #                                  scoring= scoring,
  #                                  verbose=10,
  #                                  n_jobs=-1)
  # grid_search.fit(X_train, y_train)
  pipeline.fit(X_train,y_train)

  # # find the best fitted model (the one with highest -MSE score)
  # best_model = grid_search.best_estimator_
  # print('Best params: ', grid_search.best_params_,)
  # print(grid_search.scoring)

  # calculate model scores on train and test sets (This is a model-specific score, not the -MSE anymore).
  # It uses CV to reliably calculate the prediction score on the train set.

  # cv_scores = cross_val_score(best_model, X_train, y_train, cv=CV_SPLITS)
  # train_score = cv_scores.mean()
  # print(f'train set score ({CV_SPLITS}-fold CV): {train_score:.2f}')

  test_score = pipeline.score(X_test, y_test)
  print(f'model score {scoring}: {test_score:.2f}')

  return pipeline, X_train, X_test, y_train, y_test


################################# Permutation Importance. Here, we use test set to calculate mean feature importance.
#%%
# open data
fname = '/Users/yeganeh/Codes/otka-preprocessing/docs/data_with_psds.csv'
data = pd.read_csv(fname, index_col='Unnamed: 0')
# data['subject'] = data['subject'].astype('category')
# data['hypnosis_depth'] = data['hypnosis_depth'].astype('int')


# adding new feature to the dataframe (subtraction of baseline and experience part):
# col = [col_names[i]+'-subtracted' for i in range(len(col_names))]
# new_df = pd.DataFrame(index=data.index, columns=col)
# for j in col_names:
#   new_df[j+'-subtracted'] = data[j] - data[j[:-10]+'baseline1']
# data = pd.concat([data, new_df], axis=1)


# change the hypnotic depth to a categorical variable if using Classifiers
# data['hypnosis_depth_class'] = data['hypnosis_depth'].apply(lambda x:'low' if x<5 else 'high')

data = data.query('(trial_type == "True") & (description_type == "hypnosis")')
y = data['hypnosis_depth']
feature_names = []
feature_names = [i for i in data.columns if data[i].dtype in [np.float64]]
# feature_names.append('subject')
# feature_names.append('description_type')
# feature_names.append('procedure_type')
feature_names.remove('hypnotizability_total')
# feature_names.append('trial_type')
# feature_names.append('all-theta-experience')
feature_names = [i for i in feature_names  if 'diff' in i]
feature_names = [i for i in feature_names if 'baseline' and 'induction' not in i]

# feature_names.append('hypnosis_depth')
# data[feature_names].corr()['hypnosis_depth'].sort_values().drop('hypnosis_depth').plot(kind='barh', figsize=(20,20))
# plt.savefig('corr.png')
# new_features = ['FL-gamma-experience-subtracted', 'FZ-gamma-experience-subtracted',
#        'FL-broadband-experience-subtracted', 'CL-gamma-experience-subtracted', 'CR-alpha-experience-subtracted',
      #  'PL-alpha-experience-subtracted']
# feature_names = ['hypnosis_depth','all-theta-experience-subtracted',
#        'FL-alpha-experience-subtracted', 'CR-theta-experience-subtracted',
#        'PZ-theta-experience-subtracted', 'all-alpha-experience-subtracted',
#        'CL-alpha-experience-subtracted', 'PZ-alpha-experience-subtracted',
#        'all-delta-experience-subtracted', 'CR-alpha-experience-subtracted',
#        'PR-delta-experience-subtracted', 'PR-theta-experience-subtracted',
#        'CL-theta-experience-subtracted', 'PL-theta-experience-subtracted',
#        'PZ-delta-experience-subtracted', 'FL-theta-experience-subtracted',
#        'PL-delta-experience-subtracted', 'FR-alpha-experience-subtracted',
#        'PR-alpha-experience-subtracted', 'FR-theta-experience-subtracted',
#        'PL-alpha-experience-subtracted', 'CR-delta-experience-subtracted',
#        'FZ-alpha-experience-subtracted', 'OR-theta-experience-subtracted',
#        'OR-alpha-experience-subtracted', 'OR-delta-experience-subtracted',
#        'OR-broadband-experience-subtracted', 'FZ-theta-experience-subtracted',
#        'OR-beta-experience-subtracted', 'CL-delta-experience-subtracted', ]
# feature_names = [i for i in feature_names if ('all' in i) or ('Z' in i)]
# feature_names.append('hypnosis_depth')
# data = data[feature_names]
# # [data.rename(columns={i:i[:-22]}, inplace=True) for i in data.columns if i != 'hypnosis_depth']
# bar = data.corr()['hypnosis_depth'].sort_values().drop('hypnosis_depth').plot(kind='barh', figsize=(20,20), colormap='ocean')
# title_obj = plt.title('Correlation Between Hypnosis Depth and EEG Features (based on aggregated data from all or central channels)', fontdict = {'fontsize' : 17})
# plt.setp(title_obj, color='g') 
# label_obj = plt.xlabel('Pearson Correlation', fontdict = {'fontsize' : 17})
# plt.setp(label_obj, color='g') 
# # plt.savefig('corr.png')
# plt.show()
#%%

# remove middle electrodes:
# 
# feature_names = ['CR-gamma-experience-subtracted', 'PR-gamma-experience-subtracted',
#        'OL-gamma-experience-subtracted', 'FR-gamma-experience-subtracted',
#        'FL-gamma-experience-subtracted', 'CL-gamma-experience-subtracted']

X = data[feature_names]
# model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
# model.fit(X, y)
# fit the model
model, X_train, X_test, y_train, y_test = fit_and_tune_model(X, y, CV_SPLITS=5, scoring='r2')


# # # # Permutation Importance. Here, we use test set to calculate mean feature importance.
perm_importances = permutation_importance(model, X_test, y_test, n_jobs=-1)
sorted_idx = perm_importances.importances_mean.argsort()
sorted_importances = perm_importances.importances[sorted_idx]

# # # # plot feature importances in a box plot
fig, ax = plt.subplots(figsize=(10, 20))
ax.boxplot(sorted_importances.T,
            labels=[i[:-22] for i in X_test.columns[sorted_idx]],
            vert=False,)
fig.tight_layout()
plt.savefig('permutation_importance.png')
plt.show()


# print the rmse and accuracy
y_pred = model.predict(X_test)
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print(f'R squared: {model.score(X_test, y_test)}')
# (y_pred == y_test).mean()

# %%

#%%
######################## inspect multicolinear features ##################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T)/2
np.fill_diagonal(corr, 1)

# convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X_test.columns[sorted_idx], ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
plt.show()

cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
s = [feature_names[i] for i in selected_features]

X = X[s]

######################### fit model with new features ################
# fit the model
model, X_train, X_test, y_train, y_test = fit_and_tune_model(X, y, scoring='accuracy')

# model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

perm_importances = permutation_importance(model, X_test, y_test, n_jobs=-1)

sorted_idx = perm_importances.importances_mean.argsort()
sorted_importances = perm_importances.importances[sorted_idx]

# # # # plot feature importances in a box plot
fig, ax = plt.subplots(figsize=(10, 20))
ax.boxplot(sorted_importances.T,
            labels=X_test.columns[sorted_idx],
            vert=False,)
fig.tight_layout()
plt.show()

# print('Root Mean Squared Error with features removed:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print("R squared on test data with features removed: {:.2f}".format(
#       model.score(X_test, y_test)))




perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

# %%
# [sns.displot(data[i]) for i in feature_names]

# create baseline for hypnosis change variable

# create 
# intra_induction 
# create alpha 2 band