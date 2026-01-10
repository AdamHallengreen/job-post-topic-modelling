# job-post-topic-modelling

[![Release](https://img.shields.io/github/v/release/AdamHallengreen/job-post-topic-modelling)](https://img.shields.io/github/v/release/AdamHallengreen/job-post-topic-modelling)
[![Build status](https://img.shields.io/github/actions/workflow/status/AdamHallengreen/job-post-topic-modelling/main.yml?branch=main)](https://github.com/AdamHallengreen/job-post-topic-modelling/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/AdamHallengreen/job-post-topic-modelling/branch/main/graph/badge.svg)](https://codecov.io/gh/AdamHallengreen/job-post-topic-modelling)
[![Commit activity](https://img.shields.io/github/commit-activity/m/AdamHallengreen/job-post-topic-modelling)](https://img.shields.io/github/commit-activity/m/AdamHallengreen/job-post-topic-modelling)
[![License](https://img.shields.io/github/license/AdamHallengreen/job-post-topic-modelling)](https://img.shields.io/github/license/AdamHallengreen/job-post-topic-modelling)

Research project on categorising job posts

- **Github repository**: <https://github.com/AdamHallengreen/job-post-topic-modelling/>
- **Documentation** <https://AdamHallengreen.github.io/job-post-topic-modelling/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:AdamHallengreen/job-post-topic-modelling.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version




## Notes about working on the star server

The conda environment jobpost_rapids313 with cuml for using gpu supported versions of hdbscan and umap is installed by (we no longer actually use the gpu supported versions, but this was how the environment was created)
```
conda create -n job_rapids313 -c rapidsai -c conda-forge -c nvidia rapids=25.10 python=3.13 'cuda-version=12.8'
```
I choose cuda-version 12.8 because it matches the one pytorch loads automatically
The environment can then be used to install the packages in a given environment using pip.

On the star server uv doesn't work. But instead, you can (from a computer where it does work) create a requirement.txt file using:

```
uv export --no-emit-workspace --no-dev --no-annotate --no-header --no-hashes --output-file requirements.txt
```

Then use the following pip command to install them:

```
pip install -r requirements.txt
```

You also need to tell the environment that job-post-nlp is a package, by running:

```
python -m pip install -e .
```


Downgrade kaleido so you don't need chrom (which needs the internet):
```
pip uninstall -y kaleido plotly
pip install "kaleido==0.2.1" "plotly<6"
```


 However, some packages that are installed from the web, like: `da-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/da_core_news_sm-3.8.0/da_core_news_sm-3.8.0-py3-none-any.whl`
And they need to be removed manually. This specific package can be loaded from conda using `conda install spacy-model-da_core_web_sm`. (Not relevant anymore).
Other data for packages have to manually loaded (as a zip file ) like punkt_tab from nlkt (https://www.nltk.org/data.html)
It is the loaded at the begining of prepare (you might have to adjust the path)

I also had to download paraphrase-multilingual-mpnet-base-v2 using the guide on huggingface (`https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/tree/main?clone=true`) in a off-server terminal:
\# Make sure hf CLI is installed: `pip install -U "huggingface_hub[cli]`
`hf download sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
`hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
It is then located in the cache stated by the terminal (use the one in the snapshots folder) and can be transfered to the server.


install mistral model:
smaller:
hf download bartowski/Mistral-7B-Instruct-v0.3-GGUF --include "Mistral-7B-Instruct-v0.3-Q6_K.gguf"

larger:
Just downloaded the Q8_0 version from:
https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF

download tokenizer (not used):
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir H:\jobads\installation\tokenizer --include "tokenizer.*" "tokenizer.model.v3" "tokenizer_config.json" "special_tokens_map.json"




conda create -n topicmodel312_1 python=3.12 pip -y
conda activate topicmodel312_1

pip install --no-binary=ctransformers ctransformers
pip install protobuf

On the star server uv doesn't work. But instead, you can (from a computer where it does work) create a requirement.txt file using:

```
uv export --no-emit-workspace --no-dev --no-annotate --no-header --no-hashes --output-file requirements.txt
```

Then use the following pip command to install them:

```
pip install -r requirements.txt
```
You also need to tell the environment that job-post-nlp is a package, by running:

```
python -m pip install -e .
```

Downgrade kaleido so you don't need chrom (which needs the internet):
```
pip uninstall -y kaleido plotly
pip install "kaleido==0.2.1" "plotly<6"
```


ls -lh /home/b281467@PROD.SITAD.DK/code/help/installations/llama_cpp_wheels | grep -i llama
I also had some issues where I had to force a reinstall of spacy-loggers

Since uv doesn't work I've also install precommit:
 `pip install pre-commit`
 and linked the requirements file:
 `echo 'pre-commit' >> requirements.txt`
You can the pre-commit and get ruff suggestions using:
`pre-commit run -a`

---




---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
