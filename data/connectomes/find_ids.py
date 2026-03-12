import numpy as np
ids_data = np.load('processed/mb_circuit_right_ids.npz', allow_pickle=True)
mbon_ids = ids_data['mbon_ids'].tolist()

import fafbseg
fafbseg.flywire.set_default_dataset('public')
ann = fafbseg.flywire.search_annotations(mbon_ids)

mbon11 = ann[ann['cell_type'] == 'MBON11']
mbon01 = ann[ann['cell_type'] == 'MBON01']

for _, row in mbon11.iterrows():
    idx = mbon_ids.index(row['root_id'])
    print(f"MBON11 {row['side']}: index {idx}")

for _, row in mbon01.iterrows():
    idx = mbon_ids.index(row['root_id'])
    print(f"MBON01 {row['side']}: index {idx}")