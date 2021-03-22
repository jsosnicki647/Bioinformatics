import pandas as pd
from chembl_webresource_client.new_client import new_client

import sys
sys.path.append('/usr/local/lib/python.3.7/site-packages/')

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

import lazypredict
from lazypredict.Supervised import LazyRegressor

import Functions

#ChEMBL database search
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)

#selected_target will be SARS coronavirus 3C-like proteinase
selected_target = targets.target_chembl_id[4]

#filter activities to the selected_target and standard_type IC50
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

#convert res dictionary to pandas DataFrame df and write to csv
df = pd.DataFrame.from_dict(res)
df.to_csv('bioactivity_data.csv', index=False)

#create bioactivity_class list based on standard_value
#used to subset data based on IC50 values
bioactivity_class = []
for i in df.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append('inactive')
    elif float(i) <= 1000:
        bioactivity_class.append('active')
    else:
        bioactivity_class.append('intermediate')

#subset DataFrame
selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df2=df[selection]

#convert bioactivity_class to series so that it can be concatenated
bioactivity_class = pd.Series(bioactivity_class, name='bioactivity_class')
df3 = pd.concat([df2, bioactivity_class], axis=1)
df3.to_csv('bioactivity_data_preprocessed.csv', index=False)

#get Lipinski descriptors DataFrame and append to existing DataFram
df_lipinski = Functions.lipinski(df3.canonical_smiles)
df_combined = pd.concat([df3, df_lipinski], axis=1)

#cap the standard_value at 100,000,000 to simplify pIC50 conversion
df_norm = Functions.norm_value(df_combined)

#convert IC50 value to pIC50 value
df_final = Functions.pIC50(df_norm)

#subset DataFrame to exclude intermediate bioactivity class
df_2class = df_final[df_final.bioactivity_class != 'intermediate']


#PLOTS
#Frequency of active vs. inactive bioactivity classes
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x='bioactivity_class', data=df_2class, edgecolor='black')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

#molecular weight vs. LogP descriptor
plt.figure(figsize=(5.5, 5.5))
sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('plot_MW_vs_LogP.pdf')

#box and whisker plot of pIC50 values
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
plt.savefig('plot_pIC0.pdf')

#box and whisker plot of MW
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'MW', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')
plt.savefig('plot_MW.pdf')

#box and whisker plot of LogP
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'LogP', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.savefig('plot_LogP.pdf')

#box and whisker plot of NumHDonors
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'NumHDonors', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')
plt.savefig('plot_NumHDonors.pdf')

#box and whisker plot of NumHAcceptors
plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class', y = 'NumHAcceptors', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')
plt.savefig('plot_NumHAcceptors.pdf')

#Mann Whitney U test for Lipinski descriptors, active vs. inactive populations
print(Functions.mannwhitney('pIC50',df_2class))
print(Functions.mannwhitney('NumHAcceptors', df_2class))
print(Functions.mannwhitney('NumHDonors', df_2class))
print(Functions.mannwhitney('LogP', df_2class))
print(Functions.mannwhitney('MW', df_2class))

#read in acetylcholinesterase data and save to csv
#subset the data to include on the canonical smiles and chembl_id
df4=pd.read_csv('acetylcholinesterase_04_bioactivity_data_3class_pIC50.csv')
selection = ['canonical_smiles','molecule_chembl_id']
df4_selection = df4[selection]
df4_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

#read in pubchem descriptors and drop the Name column
#set Y axis to pIC50 value
df4_X = pd.read_csv('./descriptors_output.csv')
df4_X = df4_X.drop(columns=['Name'])
df4_Y = df4['pIC50']

#create new DataFrame concatenating df4_X and df4_Y
dataset4 = pd.concat([df4_X, df4_Y], axis=1)
dataset4.to_csv('acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', index=False)
df = pd.read_csv('acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')

#set X and Y variables
X = df.drop('pIC50', axis=1)
Y = df.pIC50

#remove low-variance variables from the model and transform X
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)

#split the data 80/20 in training/test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#print and plot the different model performances
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)
print(predictions_train)
plt.clf()
plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=predictions_train.index, x="R-Squared", data=predictions_train)
ax.set(xlim=(0, 1))
plt.savefig('plot_model_performance.pdf')


print('fin')

