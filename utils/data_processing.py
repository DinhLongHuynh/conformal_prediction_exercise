import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from molvs import Standardizer


def smiles_standardizer(smi, fragment=False, isotope=False, charge=False, stereo=False, tautomer = False):
    mol = Chem.MolFromSmiles(smi)
    if fragment:
        mol = Standardizer().fragment_parent(mol, skip_standardize=False)
    if isotope:
        mol = Standardizer().isotope_parent(mol, skip_standardize=False)
    if charge:
        mol = Standardizer().charge_parent(mol, skip_standardize=False)
    if stereo:
        mol = Standardizer().stereo_parent(mol, skip_standardize=False)
    if tautomer:
        mol = Standardizer().tautomer_parent(mol, skip_standardize=False)

    std_smi = Chem.MolToSmiles(mol, canonical=True)
    return std_smi


def smiles_df_to_morgan_fps(df, smiles_column="smiles", radius=2, nBits=2048):
    fps = []
    for smi in df[smiles_column]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros(nBits, dtype=int)  # or raise an error / skip
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            arr = np.zeros((nBits,), dtype=int)
            ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps)


def calc_descriptors(smi, descriptor_list = None):
    if descriptor_list is None: 
        descriptor_list = [desc[0] for desc in Descriptors._descList]

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return calculator.CalcDescriptors(mol)
    else:
        return [None] * len(descriptor_list)
    
    
def remove_correlated_features(df, threshold=0.9):
    """
    Remove highly correlated features using a more sophisticated approach.
    Keeps the feature with higher variance when removing correlated pairs.
    """
    
    corr_matrix = df.corr().abs()
    
    features_to_remove = []
    
    # Find all high correlation pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
    
    # Sort by correlation strength (highest first)
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, j,_ in high_corr_pairs:
        feature1 = df.columns[i]
        feature2 = df.columns[j]
        
        # Skip if one of the features is already marked for removal
        if feature1 in features_to_remove or feature2 in features_to_remove:
            continue
            
        # Keep the feature with higher variance
        var1 = df[feature1].var()
        var2 = df[feature2].var()
        
        if var1 >= var2:
            features_to_remove.append(feature2)
        else:
            features_to_remove.append(feature1)
    
    df_filtered = df.drop(columns=features_to_remove)
    return df_filtered







