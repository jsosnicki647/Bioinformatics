import pandas as pd
from chembl_webresource_client.new_client import new_client

#Target search for Coronavirus
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
# print(targets)

selected_target = targets.target_chembl_id[4]
# print(selected_target)

activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)
# print(df)

df.standard_type.unique()
df.to_csv('bioactivity_data.csv', index=False)

bioactivity_class = []
for i in df.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append('inactive')
    elif float(i) <= 1000:
        bioactivity_class.append('active')
    else:
        bioactivity_class.append('intermediate')

# print(bioactivity_class)

# mol_cid = []
# for i in df.molecule_chembl_id:
#     mol_cid.append(i)

# # print(mol_cid)

# canonical_smiles = []
# for i in df.canonical_smiles:
#     canonical_smiles.append(i)

# print(canonical_smiles)

selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df2=df[selection]

# print(df2)

bioactivity_class = pd.Series(bioactivity_class, name='bioactivity_class')
df3 = pd.concat([df2, bioactivity_class], axis=1)

print(df3)