"""Load and process connectome data from FlyWire."""
import numpy as np
import pandas as pd
from fafbseg import flywire
from fafbseg.flywire import NeuronCriteria as NC


def setup_flywire(dataset='public'):
    """Set up FlyWire connection.
    
    Parameters
    ----------
    dataset : str
        'public' for public release (recommended), 'production' if you have access
    
    Notes
    -----
    Requires authentication token. Run once:
        flywire.set_chunkedgraph_secret("your_token_here")
    Get token from: https://global.daf-apis.com/auth/api/v1/user/token
    """
    # flywire.set_chunkedgraph_secret("155ef81790925555205104cb480cf567")
    flywire.set_default_dataset(dataset)


def fetch_kc_mbon_ids(side=None):
    """Fetch root IDs for Kenyon cells and MBONs from FlyWire annotations.
    
    Parameters
    ----------
    side : str, optional
        'left' or 'right' to filter by hemisphere, None for both
    
    Returns
    -------
    kc_ids : list
        Root IDs for Kenyon cells
    mbon_ids : list
        Root IDs for MBONs
    kc_ann : DataFrame
        Full KC annotations (includes cell_type, hemibrain_type, etc.)
    mbon_ann : DataFrame
        Full MBON annotations
    """
    # Search by cell_class for KCs and MBONs
    # KCs are class "Kenyon_cell" or you can search by type pattern
    if side:
        kc_ann = flywire.search_annotations(
            NC(cell_class='Kenyon_cell', side=side)
        )
        mbon_ann = flywire.search_annotations(
            NC(cell_class='MBON', side=side)
        )
    else:
        # Get all KCs - they have cell_type starting with KC
        kc_ann = flywire.search_annotations('cell_type:KC', regex=True)
        # Get all MBONs
        mbon_ann = flywire.search_annotations('cell_class:MBON')
    
    kc_ids = kc_ann['root_id'].tolist()
    mbon_ids = mbon_ann['root_id'].tolist()
    
    return kc_ids, mbon_ids, kc_ann, mbon_ann


def fetch_kc_mbon_connectivity(kc_ids=None, mbon_ids=None, 
                                proofread_only=True, min_weight=1):
    """Fetch KC→MBON connectivity directly from FlyWire.
    
    Parameters
    ----------
    kc_ids : list, optional
        KC root IDs. If None, fetches all KCs.
    mbon_ids : list, optional  
        MBON root IDs. If None, fetches all MBONs.
    proofread_only : bool
        Only include proofread neurons
    min_weight : int
        Minimum synapse count to include connection
    
    Returns
    -------
    W : ndarray, shape (n_MBONs, n_KCs)
        Weight matrix (synapse counts)
    metadata : dict
        Contains kc_ids, mbon_ids, kc_types, mbon_types
    """
    # Fetch IDs if not provided
    if kc_ids is None or mbon_ids is None:
        kc_ids, mbon_ids, kc_ann, mbon_ann = fetch_kc_mbon_ids()
    else:
        kc_ann = flywire.search_annotations(kc_ids)
        mbon_ann = flywire.search_annotations(mbon_ids)
    
    print(f"Fetching connectivity for {len(kc_ids)} KCs → {len(mbon_ids)} MBONs...")
    
    # Fetch downstream connectivity from all KCs
    # This gets all KC outputs, we'll filter to MBONs
    conn = flywire.get_connectivity(
        kc_ids,
        upstream=False,  # We want KC outputs (downstream)
        downstream=True,
        proofread_only=proofread_only,
    )
    
    # Filter to only KC→MBON connections
    mbon_set = set(mbon_ids)
    kc_to_mbon = conn[conn['post'].isin(mbon_set)].copy()
    
    # Filter by minimum weight
    kc_to_mbon = kc_to_mbon[kc_to_mbon['weight'] >= min_weight]
    
    print(f"Found {len(kc_to_mbon)} KC→MBON connections")
    
    # Build weight matrix
    n_kc = len(kc_ids)
    n_mbon = len(mbon_ids)
    W = np.zeros((n_mbon, n_kc))
    
    kc_idx_map = {kid: i for i, kid in enumerate(kc_ids)}
    mbon_idx_map = {mid: i for i, mid in enumerate(mbon_ids)}
    
    for _, row in kc_to_mbon.iterrows():
        kc_id = row['pre']
        mbon_id = row['post']
        if kc_id in kc_idx_map and mbon_id in mbon_idx_map:
            j = kc_idx_map[kc_id]
            i = mbon_idx_map[mbon_id]
            W[i, j] = row['weight']
    
    # Build metadata with cell type info
    kc_types = kc_ann.set_index('root_id')['cell_type'].to_dict()
    mbon_types = mbon_ann.set_index('root_id')['cell_type'].to_dict()
    
    metadata = {
        'kc_ids': kc_ids,
        'mbon_ids': mbon_ids,
        'kc_types': [kc_types.get(kid, 'unknown') for kid in kc_ids],
        'mbon_types': [mbon_types.get(mid, 'unknown') for mid in mbon_ids],
        'kc_annotations': kc_ann,
        'mbon_annotations': mbon_ann,
        'edge_list': kc_to_mbon,
    }
    
    return W, metadata


def fetch_dan_connectivity(mbon_ids=None, proofread_only=True):
    """Fetch DAN→MBON and MBON→DAN connectivity for plasticity modeling.
    
    Parameters
    ----------
    mbon_ids : list, optional
        MBON root IDs
    proofread_only : bool
        Only include proofread neurons
    
    Returns
    -------
    dan_to_mbon : DataFrame
        DAN→MBON connections
    mbon_to_dan : DataFrame  
        MBON→DAN connections (for recurrent loops)
    dan_ann : DataFrame
        DAN annotations
    """
    # Get DANs (PPL1 and PAM clusters)
    dan_ann = flywire.search_annotations('cell_class:DAN')
    dan_ids = dan_ann['root_id'].tolist()
    
    print(f"Found {len(dan_ids)} DANs")
    
    if mbon_ids is None:
        _, mbon_ids, _, _ = fetch_kc_mbon_ids()
    
    # DAN outputs to MBONs
    dan_conn = flywire.get_connectivity(
        dan_ids,
        upstream=False,
        downstream=True,
        proofread_only=proofread_only,
    )
    
    mbon_set = set(mbon_ids)
    dan_to_mbon = dan_conn[dan_conn['post'].isin(mbon_set)]
    
    # MBON outputs to DANs (recurrent)
    mbon_conn = flywire.get_connectivity(
        mbon_ids,
        upstream=False,
        downstream=True,
        proofread_only=proofread_only,
    )
    
    dan_set = set(dan_ids)
    mbon_to_dan = mbon_conn[mbon_conn['post'].isin(dan_set)]
    
    return dan_to_mbon, mbon_to_dan, dan_ann


def save_connectivity_cache(W, metadata, filepath):
    """Save fetched connectivity to disk for faster loading.
    
    Parameters
    ----------
    W : ndarray
        Weight matrix
    metadata : dict
        Metadata dict from fetch functions
    filepath : str
        Output path (.npz)
    """
    # Convert DataFrames to dicts for saving
    save_meta = {
        'kc_ids': np.array(metadata['kc_ids']),
        'mbon_ids': np.array(metadata['mbon_ids']),
        'kc_types': np.array(metadata.get('kc_types', [])),
        'mbon_types': np.array(metadata.get('mbon_types', [])),
    }
    
    np.savez(filepath, weights=W, **save_meta)
    
    # Also save edge list as CSV for inspection
    if 'edge_list' in metadata:
        csv_path = filepath.replace('.npz', '_edges.csv')
        metadata['edge_list'].to_csv(csv_path, index=False)
    
    print(f"Saved to {filepath}")


def load_flywire_connectivity(filepath):
    """Load cached FlyWire connectivity or fetch fresh.
    
    Parameters
    ----------
    filepath : str
        Path to .npz cache file, or 'flywire' to fetch fresh
    
    Returns
    -------
    W : ndarray, shape (n_MBONs, n_KCs)
        Weight matrix
    metadata : dict
        KC and MBON IDs, types, etc.
    """
    if filepath == 'flywire':
        # Fetch fresh from FlyWire
        setup_flywire('public')
        return fetch_kc_mbon_connectivity()
    
    if filepath.endswith('.npz'):
        data = np.load(filepath, allow_pickle=True)
        W = data['weights']
        metadata = {
            'kc_ids': data.get('kc_ids', None),
            'mbon_ids': data.get('mbon_ids', None),
            'kc_types': data.get('kc_types', None),
            'mbon_types': data.get('mbon_types', None),
        }
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        W, metadata = _process_edge_list(df)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return W, metadata


def _process_edge_list(df):
    """Convert edge list to weight matrix."""
    kc_ids = sorted(df['pre_id'].unique())
    mbon_ids = sorted(df['post_id'].unique())
    
    n_kc = len(kc_ids)
    n_mbon = len(mbon_ids)
    W = np.zeros((n_mbon, n_kc))
    
    kc_idx_map = {kid: i for i, kid in enumerate(kc_ids)}
    mbon_idx_map = {mid: i for i, mid in enumerate(mbon_ids)}
    
    for _, row in df.iterrows():
        i = mbon_idx_map[row['post_id']]
        j = kc_idx_map[row['pre_id']]
        W[i, j] = row['weight']
    
    metadata = {'kc_ids': kc_ids, 'mbon_ids': mbon_ids}
    return W, metadata


def normalize_weights(W, method='max'):
    """Normalize weight matrix."""
    if method == 'max':
        max_val = np.max(W)
        return W / max_val if max_val > 0 else W
    elif method == 'rowsum':
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return W / row_sums
    return W


def fetch_full_mb_connectivity(side=None, proofread_only=True, min_weight=1):
    """Fetch complete mushroom body circuit connectivity.
    
    Gets all pairwise connections between KCs, MBONs, and DANs.
    
    Parameters
    ----------
    side : str, optional
        'left' or 'right' hemisphere, None for both
    proofread_only : bool
        Only include proofread neurons
    min_weight : int
        Minimum synapse count
    
    Returns
    -------
    connectivity : dict
        Dictionary with keys:
        - 'kc_to_mbon': DataFrame
        - 'kc_to_dan': DataFrame  
        - 'dan_to_kc': DataFrame
        - 'dan_to_mbon': DataFrame
        - 'mbon_to_dan': DataFrame
        - 'mbon_to_mbon': DataFrame (lateral connections)
        - 'dan_to_dan': DataFrame (lateral connections)
    annotations : dict
        Dictionary with 'kc', 'mbon', 'dan' DataFrames
    ids : dict
        Dictionary with 'kc', 'mbon', 'dan' ID lists
    """
    # Fetch all neuron IDs
    print("Fetching neuron annotations...")
    
    if side:
        kc_ann = flywire.search_annotations(NC(cell_class='Kenyon_cell', side=side))
        mbon_ann = flywire.search_annotations(NC(cell_class='MBON', side=side))
        dan_ann = flywire.search_annotations(NC(cell_class='DAN', side=side))
    else:
        kc_ann = flywire.search_annotations('cell_type:KC', regex=True)
        mbon_ann = flywire.search_annotations('cell_class:MBON')
        dan_ann = flywire.search_annotations('cell_class:DAN')
    
    kc_ids = kc_ann['root_id'].tolist()
    mbon_ids = mbon_ann['root_id'].tolist()
    dan_ids = dan_ann['root_id'].tolist()
    
    print(f"Found {len(kc_ids)} KCs, {len(mbon_ids)} MBONs, {len(dan_ids)} DANs")
    
    kc_set = set(kc_ids)
    mbon_set = set(mbon_ids)
    dan_set = set(dan_ids)
    
    # Fetch all KC outputs
    print("Fetching KC connectivity...")
    kc_conn = flywire.get_connectivity(
        kc_ids,
        upstream=False,
        downstream=True,
        proofread_only=proofread_only,
    )
    kc_conn = kc_conn[kc_conn['weight'] >= min_weight]
    
    kc_to_mbon = kc_conn[kc_conn['post'].isin(mbon_set)].copy()
    kc_to_dan = kc_conn[kc_conn['post'].isin(dan_set)].copy()
    
    # Fetch all DAN outputs
    print("Fetching DAN connectivity...")
    dan_conn = flywire.get_connectivity(
        dan_ids,
        upstream=False,
        downstream=True,
        proofread_only=proofread_only,
    )
    dan_conn = dan_conn[dan_conn['weight'] >= min_weight]
    
    dan_to_kc = dan_conn[dan_conn['post'].isin(kc_set)].copy()
    dan_to_mbon = dan_conn[dan_conn['post'].isin(mbon_set)].copy()
    dan_to_dan = dan_conn[dan_conn['post'].isin(dan_set)].copy()
    
    # Fetch all MBON outputs
    print("Fetching MBON connectivity...")
    mbon_conn = flywire.get_connectivity(
        mbon_ids,
        upstream=False,
        downstream=True,
        proofread_only=proofread_only,
    )
    mbon_conn = mbon_conn[mbon_conn['weight'] >= min_weight]
    
    mbon_to_dan = mbon_conn[mbon_conn['post'].isin(dan_set)].copy()
    mbon_to_mbon = mbon_conn[mbon_conn['post'].isin(mbon_set)].copy()
    
    # Summary
    print(f"\nConnectivity summary:")
    print(f"  KC→MBON:   {len(kc_to_mbon):,} connections")
    print(f"  KC→DAN:    {len(kc_to_dan):,} connections")
    print(f"  DAN→KC:    {len(dan_to_kc):,} connections")
    print(f"  DAN→MBON:  {len(dan_to_mbon):,} connections")
    print(f"  DAN→DAN:   {len(dan_to_dan):,} connections")
    print(f"  MBON→DAN:  {len(mbon_to_dan):,} connections")
    print(f"  MBON→MBON: {len(mbon_to_mbon):,} connections")
    
    connectivity = {
        'kc_to_mbon': kc_to_mbon,
        'kc_to_dan': kc_to_dan,
        'dan_to_kc': dan_to_kc,
        'dan_to_mbon': dan_to_mbon,
        'dan_to_dan': dan_to_dan,
        'mbon_to_dan': mbon_to_dan,
        'mbon_to_mbon': mbon_to_mbon,
    }
    
    annotations = {
        'kc': kc_ann,
        'mbon': mbon_ann,
        'dan': dan_ann,
    }
    
    ids = {
        'kc': kc_ids,
        'mbon': mbon_ids,
        'dan': dan_ids,
    }
    
    return connectivity, annotations, ids


def build_weight_matrices(connectivity, ids):
    """Convert edge lists to weight matrices.
    
    Parameters
    ----------
    connectivity : dict
        From fetch_full_mb_connectivity
    ids : dict
        From fetch_full_mb_connectivity
    
    Returns
    -------
    W : dict
        Dictionary of weight matrices (postsynaptic x presynaptic):
        - 'kc_to_mbon': (n_mbon, n_kc)
        - 'dan_to_kc': (n_kc, n_dan)
        - 'dan_to_mbon': (n_mbon, n_dan)
        - 'mbon_to_dan': (n_dan, n_mbon)
        - etc.
    """
    kc_ids = ids['kc']
    mbon_ids = ids['mbon']
    dan_ids = ids['dan']
    
    n_kc = len(kc_ids)
    n_mbon = len(mbon_ids)
    n_dan = len(dan_ids)
    
    kc_idx = {kid: i for i, kid in enumerate(kc_ids)}
    mbon_idx = {mid: i for i, mid in enumerate(mbon_ids)}
    dan_idx = {did: i for i, did in enumerate(dan_ids)}
    
    def _build_matrix(edge_df, pre_idx, post_idx, n_post, n_pre):
        W = np.zeros((n_post, n_pre))
        for _, row in edge_df.iterrows():
            pre = row['pre']
            post = row['post']
            if pre in pre_idx and post in post_idx:
                i = post_idx[post]
                j = pre_idx[pre]
                W[i, j] = row['weight']
        return W
    
    W = {}
    
    # KC outputs
    W['kc_to_mbon'] = _build_matrix(
        connectivity['kc_to_mbon'], kc_idx, mbon_idx, n_mbon, n_kc
    )
    W['kc_to_dan'] = _build_matrix(
        connectivity['kc_to_dan'], kc_idx, dan_idx, n_dan, n_kc
    )
    
    # DAN outputs
    W['dan_to_kc'] = _build_matrix(
        connectivity['dan_to_kc'], dan_idx, kc_idx, n_kc, n_dan
    )
    W['dan_to_mbon'] = _build_matrix(
        connectivity['dan_to_mbon'], dan_idx, mbon_idx, n_mbon, n_dan
    )
    W['dan_to_dan'] = _build_matrix(
        connectivity['dan_to_dan'], dan_idx, dan_idx, n_dan, n_dan
    )
    
    # MBON outputs
    W['mbon_to_dan'] = _build_matrix(
        connectivity['mbon_to_dan'], mbon_idx, dan_idx, n_dan, n_mbon
    )
    W['mbon_to_mbon'] = _build_matrix(
        connectivity['mbon_to_mbon'], mbon_idx, mbon_idx, n_mbon, n_mbon
    )
    
    return W


def save_full_connectivity(connectivity, annotations, ids, prefix='mb_circuit'):
    """Save all connectivity data to disk.
    
    Parameters
    ----------
    connectivity, annotations, ids : dicts
        From fetch_full_mb_connectivity
    prefix : str
        Output file prefix
    """
    # Save edge lists as CSVs
    for name, df in connectivity.items():
        df.to_csv(f'{prefix}_{name}.csv', index=False)
    
    # Save annotations
    for name, df in annotations.items():
        df.to_csv(f'{prefix}_{name}_annotations.csv', index=False)
    
    # Save IDs
    np.savez(
        f'{prefix}_ids.npz',
        kc_ids=np.array(ids['kc']),
        mbon_ids=np.array(ids['mbon']),
        dan_ids=np.array(ids['dan']),
    )
    
    # Build and save weight matrices
    W = build_weight_matrices(connectivity, ids)
    np.savez(f'{prefix}_weights.npz', **W)
    
    print(f"Saved to {prefix}_*.csv and {prefix}_*.npz")


# Example usage
if __name__ == '__main__':
    setup_flywire('public')
    
    # Fetch everything
    conn, ann, ids = fetch_full_mb_connectivity(side='right')
    
    # Build matrices
    W = build_weight_matrices(conn, ids)
    
    print(f"\nMatrix shapes:")
    for name, mat in W.items():
        print(f"  {name}: {mat.shape}, nnz={np.count_nonzero(mat)}")
    
    # Save for later
    save_full_connectivity(conn, ann, ids, prefix='mb_circuit_right')

    