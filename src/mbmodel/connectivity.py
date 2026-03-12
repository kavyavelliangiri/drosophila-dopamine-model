"""Load and process connectome data from FlyWire."""
import numpy as np
import pandas as pd

# fafbseg is only needed for live FlyWire queries, not for loading
# cached .npz / .csv data.  Import lazily to avoid hard dependency.
flywire = None
NC = None

def _ensure_flywire():
    global flywire, NC
    if flywire is None:
        from fafbseg import flywire as _fw
        from fafbseg.flywire import NeuronCriteria as _NC
        flywire = _fw
        NC = _NC


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
    _ensure_flywire()
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
    _ensure_flywire()
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
    _ensure_flywire()
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
    _ensure_flywire()
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
    _ensure_flywire()
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

def create_random_sparse(n_pre, n_post, sparsity=0.1, w_mean=1.0):
    """Create random sparse connectivity (for testing).
    
    Useful before you have real connectome data.
    
    Parameters
    ----------
    n_pre, n_post : int
        Number of pre and postsynaptic neurons
    sparsity : float
        Fraction of possible connections (0-1)
    w_mean : float
        Mean weight
    
    Returns
    -------
    W : ndarray, shape (n_post, n_pre)
    """
    n_connections = int(n_pre * n_post * sparsity)
    W = np.zeros((n_post, n_pre))
    
    # Random connection indices
    connections = np.random.choice(n_pre * n_post, n_connections, 
                                  replace=False)
    
    for conn in connections:
        i = conn // n_pre
        j = conn % n_pre
        W[i, j] = np.random.exponential(w_mean)
    
    return W

# =====================================================================
# PN (projection neuron) connectivity — for odor pipeline (v4f)
# =====================================================================

def fetch_pn_ids(glomeruli, side='right'):
    """Fetch root IDs for olfactory PNs innervating specific glomeruli.

    Uses fafbseg ``search_annotations`` with ``NeuronCriteria(hemibrain_type=...)``
    to find projection neurons for each glomerulus.  PN hemibrain_types follow
    the pattern ``{glomerulus}_{type}PN`` (e.g. ``DC1_adPN``, ``DM2_lPN``).

    Parameters
    ----------
    glomeruli : list of str
        Glomerulus names to search for (e.g. ``['DC1', 'DL5', 'DM2']``).
    side : str
        Hemisphere: ``'right'`` or ``'left'``.

    Returns
    -------
    pn_ids : list of int
        Root IDs of matched PNs.
    pn_ann : DataFrame
        Annotations with columns ``root_id``, ``hemibrain_type``, ``glomerulus``.
    """
    _ensure_flywire()
    import re

    pn_subtypes = ['adPN', 'lPN', 'lvPN', 'vPN', 'l2PN', 'lv2PN']
    all_rows = []

    for glom in glomeruli:
        found = False
        for pt in pn_subtypes:
            htype = f'{glom}_{pt}'
            try:
                ann = flywire.search_annotations(NC(hemibrain_type=htype, side=side))
                if len(ann) > 0:
                    for _, row in ann.iterrows():
                        all_rows.append({
                            'root_id': row['root_id'],
                            'hemibrain_type': row['hemibrain_type'],
                            'glomerulus': glom,
                        })
                    found = True
            except Exception:
                pass
        # Fallback: regex search for glomerulus in hemibrain_type
        if not found:
            try:
                ann = flywire.search_annotations(
                    f'hemibrain_type:{glom}', regex=True)
                ann = ann[ann['side'] == side]
                ann = ann[ann['cell_class'].isin(['ALPN', 'PN', 'uPN'])]
                if len(ann) > 0:
                    for _, row in ann.iterrows():
                        # Parse glomerulus from hemibrain_type
                        m = re.match(r'^([A-Z][A-Z0-9a-z]+?)(?:l|m)?_',
                                     str(row['hemibrain_type']))
                        g = m.group(1) if m else glom
                        all_rows.append({
                            'root_id': row['root_id'],
                            'hemibrain_type': row['hemibrain_type'],
                            'glomerulus': g,
                        })
                    found = True
            except Exception:
                pass
        status = f"{sum(1 for r in all_rows if r['glomerulus'] == glom)} PNs" if found else "NONE"
        print(f"  {glom}: {status}")

    pn_ann = pd.DataFrame(all_rows)
    pn_ids = pn_ann['root_id'].tolist()
    print(f"Total: {len(pn_ids)} PNs across {len(glomeruli)} glomeruli")
    return pn_ids, pn_ann


def fetch_pn_to_kc_connectivity(pn_ids, kc_ids):
    """Fetch PN→KC connectivity using ``flywire.get_connectivity``.

    Same pattern as ``fetch_kc_mbon_connectivity``: get PN downstream
    connections, filter to KCs, build weight matrix.

    Parameters
    ----------
    pn_ids : list of int
        PN root IDs (presynaptic).
    kc_ids : array-like of int
        KC root IDs (postsynaptic).

    Returns
    -------
    W_PN_KC : ndarray, shape (n_kc, n_pn)
        Weight matrix (synapse counts).  Convention: ``W[post, pre]``.
    edge_df : DataFrame
        Edge list with columns ``pre``, ``post``, ``weight``.
    """
    _ensure_flywire()

    kc_set = set(int(x) for x in kc_ids)

    print(f"Fetching downstream connectivity for {len(pn_ids)} PNs ...")
    conn = flywire.get_connectivity(
        pn_ids, upstream=False, downstream=True, proofread_only=True,
    )

    # Filter to KC targets
    pn_to_kc = conn[conn['post'].isin(kc_set)].copy()
    print(f"  {len(pn_to_kc)} PN→KC connections "
          f"({pn_to_kc['pre'].nunique()} PNs → {pn_to_kc['post'].nunique()} KCs)")

    # Build weight matrix (n_kc × n_pn)
    pn_idx = {int(pid): j for j, pid in enumerate(pn_ids)}
    kc_list = [int(x) for x in kc_ids]
    kc_idx = {kid: i for i, kid in enumerate(kc_list)}

    n_pn = len(pn_ids)
    n_kc = len(kc_list)
    W = np.zeros((n_kc, n_pn))
    for _, row in pn_to_kc.iterrows():
        pre = int(row['pre'])
        post = int(row['post'])
        if pre in pn_idx and post in kc_idx:
            W[kc_idx[post], pn_idx[pre]] = row['weight']

    print(f"  W_PN_KC shape: {W.shape}, nnz: {np.count_nonzero(W)}")
    return W, pn_to_kc


def fetch_and_cache_pn_kc(glomeruli, data_dir='data/connectomes/processed',
                          side='right'):
    """Fetch PN→KC connectivity for given glomeruli and save to cache files.

    Uses fafbseg to search for PNs by ``hemibrain_type`` and
    ``flywire.get_connectivity`` for downstream connections.

    Creates:
      - ``{data_dir}/mb_circuit_right_pn_to_kc.csv``  (edge list)
      - ``{data_dir}/mb_circuit_right_pn_annotations.csv``
      - ``{data_dir}/mb_circuit_right_pn_to_kc_weights.npz``

    Parameters
    ----------
    glomeruli : list of str
        Glomerulus names (e.g. ``['DC1', 'DL5', 'DM2']``).
    data_dir : str
        Output directory for cached files.
    side : str
        Hemisphere.
    """
    import os

    setup_flywire('public')

    ids_path = os.path.join(data_dir, 'mb_circuit_right_ids.npz')
    ids_data = np.load(ids_path, allow_pickle=True)
    kc_ids = ids_data['kc_ids']

    print("Fetching PN IDs and annotations ...")
    pn_ids, pn_ann = fetch_pn_ids(glomeruli, side=side)

    print("Fetching PN→KC connectivity ...")
    W_PN_KC, edge_df = fetch_pn_to_kc_connectivity(pn_ids, kc_ids)

    # Save
    edge_path = os.path.join(data_dir, 'mb_circuit_right_pn_to_kc.csv')
    ann_path = os.path.join(data_dir, 'mb_circuit_right_pn_annotations.csv')
    npz_path = os.path.join(data_dir, 'mb_circuit_right_pn_to_kc_weights.npz')

    edge_df.to_csv(edge_path, index=False)
    pn_ann.to_csv(ann_path, index=False)
    np.savez(npz_path, weights=W_PN_KC, pn_ids=np.array(pn_ids))

    print(f"Saved: {edge_path}")
    print(f"Saved: {ann_path}")
    print(f"Saved: {npz_path}")

    return W_PN_KC, pn_ann


# Example usage
if __name__ == '__main__':
    import sys

    if '--fetch-pn' in sys.argv:
        fetch_and_cache_pn_kc()
    else:
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

    