# PANTHER: Pathway Augmented Nonnegative Tensor factorization for HighER-order feature learning

### Requirements
Code is written in Python (3.7.3) and requires PyTorch (1.0.0).

### Data
In this experiment, we have used the dataset from The Cancer Genome Atlas (TCGA), which can be downloaded at https://portal.gdc.cancer.gov/. We focus on the four most prevalent cancer types: breast cancer, colorectal cancer, lung cancer and prostate cancer.

### Analysis
To perform PANTHER analysis on germline TCGA data, run
```
CUDA_VISIBLE_DEVICES=0 python tcga_ncp_ortho_transductive.py -i6000 -r0.002 -mco2
```

The code `tcga_ncp_ortho_transductive.py` is a wrapper code that takes in two pickle files: one containing subject-by-pathway-by-gene tensor and another containing subject-by-gene (or subject-by-pathway) matrix. The file also reads in confounding variables corresponding to the subjects, calls the `NCP` class in `NCPotr.py` to perform unsupervised feature learning by jointly model genetic pathways (higher-order features) and variants (atomic features).

The meanings of the parameters are defined in `tcga_ncp_ortho_transductive.py`. This code by default uses visible GPU.

The code `reactome_proc.R` parses REACTOME pathways and links genes to these genetic pathways. This allows the code to generate the sparse subject-by-pathway-by-gene tensor from the input of sparse subject-by-gene matrix.

The code `reactome_subisomorphism.py` performs subisomorphism detection and filtering between genetic pathways.

The code `tcga_tensor.py` converts raw sparse tensor into dense tensor, performs preprocessing, applies the generalized by-patient co-occurrence counting heuristic, and generates multiple pickle files used by `tcga_ncp_ortho_transductive.py`

### Citation
```
@inproceedings{luo2021panther,
  title={PANTHER: Pathway Augmented Nonnegative Tensor factorization for HighER-order feature learning},
  author={Luo, Yuan and Mao, Chengsheng},
  booktitle={AAAI},
  year={2021}
}
```

### Contact Us
Please open an issue or contact <yuan.luo@northwestern.edu> with any questions.
