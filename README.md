# The Benefits of Balance: Variance Reduction via Measure Optimization

## Abstract

Data balancing across multiple modalities or sources is used in various forms in several foundation models (e.g., CLIP, DINO), leading to superior performance. While data balancing algorithms are often motivated by other considerations, we argue that they have an unsuspected benefit when learning with batched stochastic empirical risk minimization: variance reduction via measure optimization. We provide non-asymptotic bounds for the mean squared error of the data balancing estimator and quantify its variance reduction. We show that this reduction effect is related to the decay of the spectrum of two particular Markov operators, and that the data balancing algorithms perform measure optimization. We explain how various forms of data balancing in contrastive multimodal learning and self-supervised learning can be interpreted as instances of this variance reduction scheme.

## Background

Given an initial probability measure $R$ over $\mathcal{X} \times \mathcal{Y}$ and target marginal distributions $P_X$ on $\mathcal{X}$ and $P_Y$ on $\mathcal{Y}$, *data balancing* refers to modifying $R$ by repeatedly applying the operations
$$
    R = R_X R_{Y|X} \mapsto P_X R_{Y|X} \text{ or } R = R_Y R_{X|Y} \mapsto P_Y R_{X|Y},
$$
where $R_X$ and $R_Y$ are the marginals of $R$ and $R_{Y|X}$ and $R_{X|Y}$ are the respective conditional distributions. This codebase contains scripts and notebooks to apply this procedure in the context of standard data analysis and defining a loss for CLIP models.

## Dependencies

We recommend a hardware environment has at least 32GB CPU RAM and a GPU with at least 12GB RAM. The code runs in Python 3 with the standard Python scientific stack along with Huggingface `transformers`. These packages can be downloaded using `pip` by running
```
pip install numpy scipy pandas matplotlib seaborn transformers
```
In addition, please install PyTorch following the [installation instructions](https://pytorch.org/get-started/locally/) for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Code

The files in the repo can be used for the following purposes.

**Illustration:** The notebook `illustration.ipynb` contains a walkthrough of the balancing procecdure applied to an empirical joint distribution. It produces visuals of the joint probability mass function and the individual marginals after applying each iteration of the procedure.

**Zero-Shot Evaluation:** Evaluation was done by using [CLIP benchmark](https://github.com/LAION-AI/CLIP_benchmark) repo. First, install the package using:
```
pip install clip-benchmark
```
For background, you may read the [instructions](https://github.com/LAION-AI/CLIP_benchmark?tab=readme-ov-file#how-to-add-other-clip-models) on adding custom CLIP models. However, we provide the script that needs to be added to your CLIP Benchmark installation in order to allow zero-shot prediction. Follow these steps:
1. Place the `miniclip.py` file into `clip_benchmark/models` (this covers steps 1 and 2 of the instructions above).
2. Add the `load_miniclip` function at the bottom of the file into the `TYPE2FUNC` dictionary in `clip_benchmark/models/__init__.py`. 
3. CLIP Benchmark runs with a command line interface which downloads evaluation datasets dynamically. We have included a bash script `clip_benchmark.sh` with the commands needed. You can simply change the `root` variable to a directory that can store the downloaded data. Then, change the `model` variable to one of `joint_clip`, `orig_clip`, or `double_clip`, referring to the variants with 0, 1, or 2 iterations of balancing (see Section 4 of the manuscript).
4. Run the script to generate a `.json` output containing the results, as well as a printout. You may also add additional zero-shot evaluation datasets included in CLIP Benchmark by adding their names to the `clip_benchmark.sh` script.

**Model Architecture:** Because models are loaded within the identify files (e.g. `miniclip.py`), all model definitions and base embeddings are in this file. The saved files refer to the "head" models. 

**Data:** The subset of ImageNet-Captions used is listed in `imagenet_captions_train_c250.csv` with a column for the filename of the ImageNet image and a column for the associated caption. Because the captions are included in the file, you can simply retrieve the images from the [ImageNet](https://www.image-net.org/download.php) dataset directly.