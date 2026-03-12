# In your connectomes directory
from mbmodel.connectivity import setup_flywire, fetch_full_mb_connectivity, build_weight_matrices, save_full_connectivity

setup_flywire('public')
conn, ann, ids = fetch_full_mb_connectivity(side='right')  # right hemisphere only
W = build_weight_matrices(conn, ids)
save_full_connectivity(conn, ann, ids, prefix='processed/mb_circuit_right')