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

On the star server uv doesn't work. But instead, you can (from a computer where it does work) create a requirement.txt file using:

```
uv export --no-emit-workspace --no-dev --no-annotate --no-header --no-hashes --output-file requirements.txt
```

Which can then be used to install the packages in a given environment using pip. However, some packages that are installed from the web, like: `da-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/da_core_news_sm-3.8.0/da_core_news_sm-3.8.0-py3-none-any.whl`
And they need to be removed manually. This specific package can be loaded from conda using `conda install spacy-model-da_core_web_sm`. (Not relevant anymore).
Other data for packages have to manually loaded manually (as a zip file ) like punkt_tab from nlkt (https://www.nltk.org/data.html)

I also had to download paraphrase-multilingual-mpnet-base-v2 using the guide on huggingface (`https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/tree/main?clone=true`):
\# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download sentence-transformers/paraphrase-multilingual-mpnet-base-v2
It is then located in the cache stated by the terminal (use the one in the snapshots folder) and can be transfered to the server.

How to install chromium and dependecies (what I did, all steps might not be necesary )
```
conda install -c esss chromium
conda install -c esss nss  libcups xorg-libxcomposite xorg-libxdamage \
               xorg-libxfixes xorg-libxrandr  pango cairo alsa-lib xkbcommon mesa-libgbm atk-bridge where not found
conda install -c conda-forge \
  libxkbcommon libgbm alsa-lib nss cups pango cairo atk-1.0 at-spi2-atk \
  xorg-libxcomposite xorg-libxdamage xorg-libxfixes xorg-libxrandr
conda install -c conda-forge xorg-libxscrnsaver

conda install -c conda-forge libXss

```
xkbcommon mesa-libgbm atk-bridge where not found in the first
cups libgbm not found in the second
I also used `!ldd chromepath | grep "not found" || true`
to check missing depdendencies
where chrome path can be chromepath can be found in python: `import shutil; shutil.which("chrome")`

Then use the following pip command to install the rest:

```
pip install -r requirements.txt
```

You also need to tell the environment that job-post-nlp is a package, by running:

```
python -m pip install -e .
```

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
