## Welcome to DeepMoDRP's Reimplementation Documentation!
This repository corresponds to the reimplementation in the paper titled "DeepMoDRP: A Multi-Omics-Based Deep Learning Framework for Drug Response Prediction in Brain Cancer".

<img width="1155" height="699" alt="Final poster diagram" src="https://github.com/user-attachments/assets/aef145ff-f4bc-4153-987a-8f9f0dd79fb4" />
Figure 1: Overview of the DeepMoDRP framework

## Resources
- [Original Paper](https://onlinelibrary.wiley.com/doi/10.1002/minf.70020)
- [DOI Reference](https://doi.org/10.1002/minf.70020)

## Installation
```bash
python dae_ae_cnv.py
```

1. Download the dae_ae_cnv.py, dae_ae_cnv.py, functions.py, gnn_drug.py, pre_process.py
2. Run dae_ae_cnv.py and dae_ae_rna.py to compress the data before preprocessing
3. Run pre_process.py to preprocess the data before feeding it to the DRP model
4. Run deepmodrp.py to run the DRP model and get the final results

## Data Processing Steps
```bash
python dae_ae_cnv.py
```

| Step | Scripts |
|------|---------|
| 1 | '' |
| 2 | '' |

## Preprocessed dataset loading
| Dataset Module | Usage |
|----------------|-------|
| 1 | '' |
| 2 | '' |


## Training and testing the developed models
| Model Module | Usage |
|--------------|-------|
| 1 | '' |

## Enrichment Analysis
Predicted drug responses can be further analyzed through Gene Ontology (GO) and Kyoto Encyclopedia of Genes and Genomes (KEGG) enrichment analyses. We created a .csv file containing the top 20% lowest IC50 predictions made by our model. We did this to find out whether the genes and biological pahtways associated with our model's predictions correlate with already established cancer-related pathways.
The workflow was:
- Select highly ranked predicted drug cell line responses and save it as top20_predicted_drug_reponses.csv
- Collated a drug list from reponses and identify biological targets of these drugs and saved it as drug_ids.csv
- Collated a 
Perform GO enrichment analysis.
Identify enriched biological pathways.

## Authors
- [IyiOluwa Adaramola](adaram_i1@denison.edu) - Computer Science, Denison University
- [Anowarul Kabir](akabir@usf.edu) - Bellini College of AI, Cybersecurity & Computing, University of South Florida

## Licencse
MIT License

Copyright (c) 2026 Kabir's AI to Bio (A2B) Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

