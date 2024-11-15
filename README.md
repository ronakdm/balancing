# The Benefits of Balance: From Information Projections to Variance Reduction

This repository contains code and experiments for "The Benefits of Balance: From Information Projections to Variance Reduction" (NeurIPS '24). Please find instructions on software/hardware dependencies, reproducing all results from the manuscript below, and additional illustrations below.

## Abstract

Data balancing across multiple modalities or sources is used in various forms in several foundation models (e.g., CLIP, DINO), leading to superior performance. While data balancing algorithms are often motivated by other considerations, we argue that they have an unsuspected benefit when learning with batched stochastic empirical risk minimization: variance reduction via measure optimization. We provide non-asymptotic bounds for the mean squared error of the data balancing estimator and quantify its variance reduction. We show that this reduction effect is related to the decay of the spectrum of two particular Markov operators, and that the data balancing algorithms perform measure optimization. We explain how various forms of data balancing in contrastive multimodal learning and self-supervised learning can be interpreted as instances of this variance reduction scheme.

## Background

Given an initial probability measure $R$ over $\mathcal{X} \times \mathcal{Y}$ and target marginal distributions $P_X$ on $\mathcal{X}$ and $P_Y$ on $\mathcal{Y}$, *data balancing* refers to modifying $R$ by repeatedly applying the operations

$$
    R = R_X R_{Y|X} \mapsto P_X R_{Y|X} \text{ or } R = R_Y R_{X|Y} \mapsto P_Y R_{X|Y},
$$

where $R_X$ and $R_Y$ are the marginals of $R$, while $R_{Y|X}$ and $R_{X|Y}$ are the respective conditional distributions. In the paper, we describe how this procedure lies at the heart of common self-supervised learning (SSL) approaches such as self-labeling and constrastive learning. This codebase contains scripts and notebooks to apply this procedure in the context of both standard data analysis and CLIP training by modifying the loss function.

## Quickstart

The method described above is in fact very simple to implement, and can be contained in a single code snippet. The existence of this repo is primarily for integrating it into existing pipelines for training and benchmarking CLIP models. See the following Numpy implementation below.
```
def data_balance(pmat, px, py, num_iter):
    """
        pmat: m-by-l matrix representing the initial probability mass function for X (taking one of m values) and Y (taking one of l values).
        px: m-sized array containing the desired X marginal.
        px: l-sized array containing the desired Y marginal.
        num_iter: number of balancing iterations, where each iteration includes both the X and Y steps.
    """
    if np.sum(np.sum(pmat, axis=1) == 0) + np.sum(np.sum(pmat, axis=0) == 0) > 0:
        raise RuntimeError(
            "Missing mass in this sample. Try a larger sample size.")
        
    est = [pmat.copy()]
    for i in range(1, num_iter):
        pmat = (px / np.sum(pmat, axis=1)).reshape(-1, 1) * pmat
        pmat = pmat * (py / np.sum(pmat, axis=0))
        est.append(pmat.copy())
    return est
```
This is applied to simulated data in `notebooks/simulated_data.ipynb`.

## Dependencies

We recommend a hardware environment has at least 32GB CPU RAM and a GPU with at least 12GB RAM for ease of use. The code runs in Python 3 with the standard Python scientific stack along with PyTorch and packages built on top of it.
These packages can be downloaded using `pip` by running:
```
pip install numpy scipy pandas matplotlib seaborn transformers
```
Next, please install PyTorch following the [installation instructions](https://pytorch.org/get-started/locally/) for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
Finally, we reply on Huggingface `transformers` (installed via `pip install transformers`), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark). For the latter two, see the links for up-to-date installation information.

## Code

The files in the repo can be used for the following purposes.

**Direct Reproduction:** To reproduce the main experimental figures from the paper, use the following files in the `notebooks` folder.

| Figure      | File |
| ----------- | ----------- |
| 2   | `figure_zero_shot.ipynb`  |
| 3   | `figure_marginals.ipynb`  |
| 5   | `figure_metaclip.ipynb`   |

The figures are produced using the zero-shot evaluation results that are in the `results` folder and the models in the `models` folder. 
To go further and recreate the models and data, see `notebooks/create_imagenet_captions.ipynb` to see how the embeddings are generated. 
Similarly, the MetaCLIP data curation is carried out in the `notebooks/create_metaclip_dataset.ipynb`.
The remaining sections show how to reproduce these results step-by-step.

**Evaluations:** Evaluation was done by using [CLIP benchmark](https://github.com/LAION-AI/CLIP_benchmark) repo. 
The results are saved in `results/` but they can be recreated (along with other evaluations) using the following procedure.
First, install the package using:
```
pip install clip-benchmark
```
For background, you may read the [instructions](https://github.com/LAION-AI/CLIP_benchmark?tab=readme-ov-file#how-to-add-other-clip-models) on adding custom CLIP models. However, we provide the script that needs to be added to your CLIP Benchmark installation in order to allow zero-shot prediction. Follow these steps:
1. Change the `MODEL_DIR` variable in the `miniclip.py` script to the absolute path where the `balancing/models` folder lives on your machine.
2. Place the `miniclip.py` file into `clip_benchmark/models` folder in your particular installation of `clip_benchmark` (this covers steps 1 and 2 of the instructions above).
3. Add the `load_miniclip` function at the bottom of the file into the `TYPE2FUNC` dictionary in `clip_benchmark/models/__init__.py`. 
4. CLIP Benchmark runs with a command line interface which downloads evaluation datasets dynamically. We have included bash scripts in `scripts` with the commands needed for each task: **zero-shot classification**, **zero-shot retrieval**, and **linear probing**. You can simply change the `root` variable to a directory that can store the downloaded data. Then, change the `model` variable to one of `joint_clip`, `orig_clip`, or `double_clip`, referring to the variants with 0, 1, or 2 iterations of balancing (see Section 4 of the manuscript).
5. Run the script to generate a `.json` output containing the results, as well as a printout. You may also add additional evaluation datasets included in CLIP Benchmark by adding their names to the corresponding `scripts/{task}.sh` script.

**Model Architecture:** Because models are loaded within the identify files (e.g. `miniclip.py`), all model definitions and base embedding models are copied in this file. 
Thus, only one file is needed to perform the custom evaluations in the previous step. The saved files refer to the "head" models. The architecture (without any OpenCLIP code) is stored in `src/multimodal_models.py`.

**Data:** The subset of ImageNet-Captions used is listed in `imagenet_captions_train_c250.csv` with a column for the filename of the ImageNet image and a column for the associated caption. Because the captions are included in the file, you can simply retrieve the images from the [ImageNet](https://www.image-net.org/download.php) dataset directly.

## References

If you found this repository useful, please consider citing the following paper.

```
@inproceedings{Liu2024TheBenefits,
  title={{The Benefits of Balance: From Information Projections to Variance Reduction}},
  author={Liu, Lang and Mehta, Ronak and Pal, Soumik and Harchaoui, Zaid},
  booktitle={NeurIPS},
  year={2024}
}
```