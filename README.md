<img src="easydecon-logo.png" alt="Logo" width=130 style="vertical-align: middle; margin-right: 10px;"/>  

[![PyPI version](https://badge.fury.io/py/easydecon.svg)](https://badge.fury.io/py/easydecon)  
A package to analyze celltypes on high definition spatial profiling assays

Installation
------------
It is recommended to install the package in a virtual environment or a Conda environment. To create a Conda environment, run the following command:

```bash
conda create -n easydecon python=3.10.14
conda activate easydecon
```

You can install from PyPi:

```bash
pip install easydecon
```

To install directly from GitHub using pip into the active environment, run the following command:

```bash
pip install git+https://github.com/sinanugur/easydecon.git
```

Overview
--------
<img src="easydecon-overview.png" alt="Worfklow Overview"/>

Absolute Minimal Example
---------------
```python
from easydecon.easydecon import *
from easydecon.config import *
from easydecon.extra import *

#read your DESeq table into a markers_df
#sdata is your VisiumHD file in SpatialData format or segmented AnnData object, assumed you QC and etc.
markers_df=read_markers_dataframe(sdata,filename="scanpy_deseq_table.csv")

#run easydecon
ph1, ph2, assigned_labels, posterior_df, proportions_df= easydecon_workflow(sdata,markers_df=markers_df)

#or setting prior genes
ph1, ph2, assigned_labels, posterior_df, proportions_df= easydecon_workflow(sdata,markers_df=markers_df,marker_genes=["gene1","gene2","gene3"])

`assigned_labels` will be added to sdata.obs and contain the celltype assignments
`proportions_df` will contain the estimated proportions for each celltype

```

Usage and Documentation
-----------------------
You may find our example notebooks in the `notebooks` folder.

- Demo notebook for a single-cell Anndata object (demo)[https://github.com/sinanugur/easydecon/blob/main/notebooks/demo.ipynb]
- Demo notebook for macrophage markers (demo_macrophage)[https://github.com/sinanugur/easydecon/blob/main/notebooks/demo_macrophage.ipynb]
- Minimal example notebook (minimal)[https://github.com/sinanugur/easydecon/blob/main/notebooks/demo_minimal_example.ipynb]
