# drosophila-dopamine-model

A hybridized computational model of the *Drosophila melanogaster* mushroom body (MB) that combines:

- **Rate-model network dynamics** from [Huang et al. (2024)](https://doi.org/10.1371/journal.pcbi.1012551) — linear ODEs for KC, DAN, and MBON populations.
- **Dopaminergic Plasticity Rule (DPR)** from [Gkanias et al. (2022)](https://doi.org/10.7554/eLife.75132) — two-component (potentiation/depression) dopamine signal driving KC→MBON synaptic weight updates.
- **FlyWire connectome data** for biologically accurate KC→MBON and DAN→MBON connectivity.

---

## Model Overview

### Network dynamics (Huang et al., 2024)

At each timestep the three populations are updated via:

**KCs** (linear activation, eq 3.1):
```
τ_KC · dΔx_KC/dt = w_odor(t) · x_odor(t) − Δx_KC
```

**DANs** (linear activation, eq 3.2):
```
τ_DAN · dΔx_DAN/dt = w_punish · x_punish(t)
                    + Σ_i w_KD,ij · Δx_KC,i
                    + Σ_l w_MD,lj · Δx_MBON,l
                    − Δx_DAN
```

**MBONs** (piecewise-linear activation, eq 3.3):
```
τ_MBON · dΔx_MBON/dt = f_a( Σ_i w_KM,ij(t) · Δx_KC,i
                              + Σ_l w_MM,lj · Δx_MBON,l )
                        − Δx_MBON

f_a(x) = clip(x, 0, M_MBON)
```

### Sensory adaptation (Huang et al., 2024, eq 2.1)

ORN-level odor weight adapts during continuous stimulation:
```
dw_odor/dt = −x_odor / τ_adapt  · w_odor
           + (1 − x_odor) / τ_recover · (A_odor − w_odor)
```

### Dopaminergic Plasticity Rule (Gkanias et al., 2022)

Two dopamine components track the DA signal at different timescales:
```
τ_short · dD_△/dt = d_j(t) − D_△_j        (fast; potentiation)
τ_long  · dD_▽/dt = d_j(t) − D_▽_j        (slow; depression)
```

Net dopaminergic factor:
```
δ_j(t) = D_△_j(t) − D_▽_j(t)
```

KC→MBON weight update:
```
dW_ij/dt = η · δ_j(t) · [ k_i(t) + W_ij(t) − w_rest ]
```

---

## Repository Structure

```
drosophila-dopamine-model/
├── src/mbmodel/
│   ├── models.py        # NeuronPopulation, MushroomBodyNetwork
│   ├── plasticity.py    # GkaniasDPR, DopaminePlasticity
│   ├── stimuli.py       # OdorAdaptation, create_sparse_odor
│   ├── connectivity.py  # FlyWire connectome loading / building matrices
│   └── utils.py         # relu, euler_step, Recorder
├── experiments/
│   ├── v1_test.py              # minimal KC/MBON/DAN rate dynamics test
│   ├── v3_randomodormodel.py   # full-scale training + weight-change analysis
│   └── configs/config.yaml     # model hyperparameters
├── data/connectomes/           # FlyWire connectome data (not tracked)
├── results/                    # output figures
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone https://github.com/kavya-velliangiri/drosophila-dopamine-model.git
cd drosophila-dopamine-model
pip install -e .
```

---

## Usage

### Minimal example (no connectome data)

```python
import numpy as np
from mbmodel.models import MushroomBodyNetwork

config = dict(
    n_KCs=100, n_MBONs=10, n_DANs=5,
    dt=0.1,
    tau_KC=10.0, tau_MBON=20.0, tau_DAN=15.0,
    r_max_KC=100.0, r_max_MBON=100.0, r_max_DAN=100.0,
    learning_rate=0.01, w_rest=1.0,
    tau_short=200.0, tau_long=2000.0,
)

net = MushroomBodyNetwork(config)

odor = np.random.choice([0.0, 50.0], size=100, p=[0.9, 0.1])

for step in range(10000):
    t = step * config['dt']
    I_odor = odor if 500 <= t < 1000 else np.zeros(100)
    x_punish = 1.0 if 1500 <= t < 1600 else 0.0
    net.step(I_odor, x_punish)

print(net.get_weight_change().mean())
```

### With sensory adaptation

```python
from mbmodel.stimuli import OdorAdaptation

adaptor = OdorAdaptation(A_odor=50.0, tau_adapt=1000.0, tau_recover=3000.0)

for step in range(10000):
    t = step * 0.1
    x_odor = 1.0 if 500 <= t < 1000 else 0.0
    w_odor = adaptor.step(x_odor, dt=0.1)
    I_odor = w_odor * x_odor
    net.step(I_odor)
```

### Run experiment scripts

```bash
python experiments/v1_test.py
python experiments/v3_randomodormodel.py
```

---

## Parameters (experiments/configs/config.yaml)

| Parameter | Default | Description |
|---|---|---|
| `tau_KC` | 10 ms | KC decay time constant |
| `tau_MBON` | 20 ms | MBON decay time constant |
| `tau_DAN` | 15 ms | DAN decay time constant |
| `learning_rate` η | 0.01 | DPR learning rate |
| `w_rest` | 1.0 | Resting KC→MBON weight |
| `tau_short` | 200 ms | Fast DA filter (D_△) |
| `tau_long` | 2000 ms | Slow DA filter (D_▽) |

---

## References

- Huang, Y. et al. (2024). Dopamine-mediated interactions between short- and long-term memory dynamics. *PLOS Computational Biology*.
- Gkanias, E. et al. (2022). An incentive circuit for memory dynamics in the mushroom body of *Drosophila melanogaster*. *eLife*.
