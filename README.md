# Welcome to DeepMoDRP's Reimplementation Documentation!
This repository corresponds to the reimplementation in the paper titled "DeepMoDRP: A Multi-Omics-Based Deep Learning Framework for Drug Response Prediction in Brain Cancer". 
<img width="1155" height="699" alt="Final poster diagram" src="https://github.com/user-attachments/assets/aef145ff-f4bc-4153-987a-8f9f0dd79fb4" />
Figure 1: Overview of the DeepMoDRP framework

## Other Resources
- [Original Paper](https://onlinelibrary.wiley.com/doi/10.1002/minf.70020)
- [DOI Reference](https://doi.org/10.1002/minf.70020)
- [Poster presentation]()
- [Project Description](https://github.com/kabir-ai2bio-lab/MultiomicsInformedDrugResponse/blob/main/project_description.docx) (A more detailed description of research project)

## Installation
### Clone Repository
```bash
git clone https://github.com/kabir-ai2bio-lab/MultiomicsInformedDrugResponse.git
cd DeepMoDRP
```

### Create virtual environment
Using conda:
```bash
conda create -n deepmodrp python=3.10 -y
conda activate deepmodrp
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Deactive and Remove Environment
```bash
# deactivation
conda deactivate
# removal
conda remove --name deepmodrp --all
```

## Data Preprocessing Steps
The ```pre_process``` directory contains scripts for transforming raw datasets into model-ready inputs.

The preprocessing pipeline performs:
| Step | Description                                                 |
| ---- | ----------------------------------------------------------- |
| 1    | Load raw multi-omics datasets                               |
| 2    | Match cell lines across datasets                            |
| 3    | Normalize RNA, CNV, MET, and MUT features                   |
| 4    | Encode drug structures into molecular graphs                |
| 5    | Split datasets into training, validation, and testing folds |
| 6    | Save processed datasets as pickle files                     |

After preprocessing, the generated datasets are stored in:
```bash
datasets/
└── cross-val/
    ├── train_fold_0.pkl
    ├── validation_fold_0.pkl
    ├── test_fold_0.pkl
    ├── train_fold_1.pkl
    ├── validation_fold_1.pkl
    ├── test_fold_1.pkl
    ...
```

## Model Architecture
The model is implemented in:
```bash
model/
└── deepmodrp.py
```
The architecture consists of three major components:
1. Multi-omics Feature Representation
2. Drug Molecular Representation
3. Drug Response Prediction

## Training and testing the developed models
The complete pipeline can be executed using:
```bash
python example_run.py
```
Hyperparameters can also be modified in ```example_run.py```

The script performs:
1. Data preprocessing
2. Model training
3. Validation using early stopping
4. Testing on unseen data
5. Reporting evaluation metrics

### Evaluation Metrics
The model reports:
| Metric | Description                     |
| ------ | ------------------------------- |
| RMSE   | Root Mean Square Error          |
| PCC    | Pearson Correlation Coefficient |
| R²     | Coefficient of Determination    |
| MAPE   | Mean Absolute Percentage Error  |

Example output:
```
Results from testing, validation and evaluating model

RMSE: 0.0479
PCC: 0.9448
R²: 0.8918
MAPE: 11.02
```


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

