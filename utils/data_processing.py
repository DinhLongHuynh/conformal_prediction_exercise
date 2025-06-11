import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from molvs import Standardizer


def smiles_standardizer(smi, fragment=False, isotope=False, charge=False, stereo=False, tautomer = False):
    """
    Standardize a SMILES string using various molecular standardization procedures.
    
    This function applies standardization procedures to a SMILES string
    using the molvs library. Each standardization step can be optionally enabled
    to handle different molecular variations and ensure consistent representation.
    
    Parameters
    ----------
    smi : str
        Input SMILES string
    fragment : bool, default=False
        If True, select the largest molecular fragment (useful for removing salts and counterions).
    isotope : bool, default=False
        If True, remove isotope labels.
    charge : bool, default=False
        If True, neutralize the molecule.
    stereo : bool, default=False
        If True, remove stereochemistry information.
    tautomer : bool, default=False
        If True, select the canonical tautomer form.
    
    Returns
    -------
    str
        Canonical SMILES string after applying the selected standardization procedures.
    """

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
    """
    Convert SMILES strings in a DataFrame to Morgan fingerprint array.
    
    This function generates Morgan fingerprints for molecules represented
    as SMILES strings in a pandas DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing SMILES strings.
    smiles_column : str, default="smiles"
        Name of the column containing SMILES strings.
    radius : int, default=2
        Radius parameter for Morgan fingerprint generation.
    nBits : int, default=2048
        Length of the fingerprint bit vector.
    
    Returns
    -------
    numpy.ndarray of shape (n_molecules, nBits)
        2D array where each row represents a molecule's Morgan fingerprint as a
        binary vector. Invalid SMILES strings result in zero vectors.
    """

    fps = []
    for smi in df[smiles_column]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros(nBits, dtype=int)
            print(f"Warning: Invalid SMILES string skipped: {smi}")
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            arr = np.zeros((nBits,), dtype=int)
            ConvertToNumpyArray(fp, arr) # Speed up process
        fps.append(arr)
    return np.array(fps)


def calc_descriptors(smi, descriptor_list = None):
    """
    Calculate molecular descriptors for a given SMILES string.
    
    This function computes a comprehensive set of molecular descriptors using RDKit's
    descriptor calculation framework.
    
    Parameters
    ----------
    smi : str
        Input SMILES string for which to calculate descriptors.
    descriptor_list : list of str, optional
        List of descriptor names to calculate. If None, calculates all available
        descriptors from RDKit's descriptor list.
    
    Returns
    -------
    tuple or list
        Tuple containing calculated descriptor values in the same order as
        descriptor_list. Returns a list of None values (same length as
        descriptor_list) if the SMILES string is invalid.
    """

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
    Remove highly correlated features using a variance-based selection strategy.
    
    This function identifies pairs of features with correlation coefficients above
    the specified threshold and removes one feature from each correlated pair.
    When deciding which feature to remove, it keeps the feature with higher variance,
    as this typically contains more information.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numerical features.
    threshold : float, default=0.9
        Correlation threshold above which features are considered highly correlated.
        Must be between 0 and 1.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with highly correlated features removed. The returned DataFrame
        maintains the same index as the input but with fewer columns.
    """
    
    corr_matrix = df.corr().abs()
    features_to_remove = []
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
    
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







