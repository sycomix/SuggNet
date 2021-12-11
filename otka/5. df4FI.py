# %%
# 
import os.path as op
import pandas as pd
import numpy as np
from pandas.io import feather_format 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import eli5
from eli5.sklearn import PermutationImportance
import shap

#%%
data_dir = '/Users/yeganeh/Codes/otka-preprocessing/'
fname = op.join(data_dir, 'data_with_hypnotizability.xlsx')

data = pd.read_excel(fname, header=1)
data.set_index('index', inplace=True)

#%%
# prepare data for the training
# initiate an empty dataframe
a = data.index
index = [x for pair in zip(a,a,a,a) for x in pair]
columns = []
for i in range(len(data.columns.tolist())):
  if data.columns.tolist()[i][-1] == '1' and data.columns.tolist()[i][0:4] != 'time' and data.columns.tolist()[i][0:3] != 'tsz':
    columns.append(data.columns.tolist()[i])
columns = [x.replace('_1','') for x in columns]

df_ = pd.DataFrame(index=index, columns=columns)
df_ = df_.fillna(0)

# 
temp = []
sub_ids = data.index.tolist()
for i in sub_ids:
  for j in range(len(columns)):
    for z in range(1,5):
      temp.append(data.loc[i,f'{columns[j]}_{z}'])
    df_.loc[i,columns[j]] = temp
    temp = []

# add other variables to this new dataset
numeric_col = ['attitude_towards_hypnosis', 'effective_of_hypnoanalgesia', 'knowledge_level_on_hypnosis',
               'motivation_to_get_hypnotized', 'hypnotizability_total']
df_2 = data[numeric_col]
for i in range(len(df_2.columns.tolist())):
  df_[df_2.columns.tolist()[i]] = df_2[df_2.columns.tolist()[i]].repeat(4)
# df_.procedure_type = df_.procedure_type.astype('category').cat.codes
# df_.description_type = pd.get_dummies(df_.description_type).values.astype('float64')
# df_.trial_type = df_.trial_type.astype('category').cat.codes
# df_.drop(columns='experiences_freetext', inplace=True)
# df_.head()

# %%
# open two psd dfs and merge them into one
psd_df = pd.read_csv('psd_df.csv')
psd_df.rename(columns={'Unnamed: 0.1':'index'}, inplace=True)
psd_df.set_index('index', inplace=True)
psd_df.drop(columns='Unnamed: 0', inplace=True)

# transform the power values to dB
df_new = pd.DataFrame(index=psd_df.index , columns=psd_df.columns)
for column in psd_df.columns:
    for item in psd_df.index:
        df_new.loc[item, column] = (10. * np.log10(psd_df.loc[item,column]))

#%%
# Change the shape of psd df
# initiate a df with desired shape
columns = []
for task in ['induction', 'experience']:
    for item in df_new.columns:
        columns.append(item+'-'+task)

index = []
for item in df_new.index:
    for trial in ['trial1', 'trial2', 'trial3', 'trial4']:
        index.append(item[:6]+'-'+trial)
index = list(dict.fromkeys(index))

df_0 = pd.DataFrame(index=index, columns=columns)

for ind in df_new.index:
    for column in df_new.columns:
        if ind[7:-1] == 'baseline':
            continue
        else:
            df_0.loc[ind[:6]+'-'+'trial'+ind[-1], column+'-'+ind[7:-1]] = df_new.loc[ind, column]
#%%
import pickle
a_file = open("ids_map_4fi.pkl", "rb")
ids_map = pickle.load(a_file)
a_file.close()
ids_map.pop('214611')
#%%
a = list(ids_map.keys())
a.remove('21060814')
a.append('216814')
list0 = [x for pair in zip(a,a,a,a) for x in pair]
list0 = [int(i) for i in list0]
df_0['index'] = list0
df_0.set_index('index', inplace=True)
a = [int(i) for i in a]
df_new = df_.loc[a]
df_psd = pd.concat([df_new, df_0], axis=1)

df_psd.to_csv('df_4FeatureImportance.csv')


# %%
# fit model using RandomForest algorithm
# df_ = df_.dropna()
y = df_psd['hypnosis_depth']
feature_names = [i for i in df_psd.columns if df_psd[i].dtype in [('O')]]
[feature_names.remove(i) for i in ['description_type','experiences_freetext','procedure_type','trial_type']]

# ['all-alpha-experience','PL-theta-induction', 'PR-theta-induction', 'PR-alpha-induction', 'PL-alpha-induction', 'all-theta-experience']


# feature_names = numeric_col
X = df_psd[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestClassifier(n_estimators=20)
model.fit(train_X, train_y)
y_pred = model.predict(val_X)

# %%
from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(val_y, y_pred)))

# %%
perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


# %%
# compare the power in the first segment of the induction with the power in the last segment of the induction

# try feature importance with linear regression