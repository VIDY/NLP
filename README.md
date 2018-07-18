# Vidy NLP Protocol
Python implementation of the NLP models used by the contextual ad placement platform

<p align="center"><img width="70%" src="images/nlp.jpg" /></p>

Vidy has invented the first single-page invisible embed layer for video, run on the Ethereum blockchain.
With just a hold, users can now reveal tiny hyper-relevant videos hidden behind the text of any page on the web, unlocking a whole new dimension to the internet.

## Models
- [Category Detection](#category-model) 
- [Offensive Content](#safety-model)
- [Sentiment Analysis](#sentiment-analysis)
- [Keyword Extraction](#keyword-extraction)
- [Related Keywords](#related-keywords)


## Getting Started
- [Installing TensorFlow](#installing-tensorflow) 
- [Downloading models](#downloads)
- [Setting up Flask](#flask)
- [Training](#training)
- [Launching the services](#services)
- [Calling the apis](#apis)


## Installing TensorFlow

<p align="center"><img width="70%" src="images/tensorflow.png" /></p>

TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence Research organization for the purposes of conducting machine learning and deep neural networks research.

### Installation
 For succinctness we'll focus on installation instructions for the Ubuntu operating system. 
 *See [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions on how to install release binaries or how to build from source on other platforms.*


 **1. Install Python, pip, and virtualenv.**

 On Ubuntu, Python is automatically installed and pip is usually installed. Confirm the python and pip versions:

  ```shell
  $ python -V  # or: python3 -V
  $ pip -V     # or: pip3 -V
  ```

  To install these packages on Ubuntu:

  ```shell
  $ sudo apt-get install python-pip python-dev python-virtualenv   # for Python 2.7
  $ sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
  ```

  We recommend using pip version 8.1 or higher. If using a release before version 8.1, upgrade pip:

  ```shell
  $ sudo pip install -U pip
  ```

  If not using Ubuntu and setuptools is installed, use easy_install to install pip:

  ```shell
  $ easy_install -U pip
  ```

 **2. Create a directory for the virtual environment and choose a Python interpreter.**

  ```shell
  $ mkdir ~/tensorflow  # somewhere to work out of
  $ cd ~/tensorflow
  # Choose one of the following Python environments for the ./venv directory:
  $ virtualenv --system-site-packages venv             # Use python default (Python 2.7)
  $ virtualenv --system-site-packages -p python3 venv # Use Python 3.n
  ```

 **3. Activate the Virtualenv environment.**

  ```shell
  $ source ~/tensorflow/venv/bin/activate      # bash, sh, ksh, or zsh
  $ source ~/tensorflow/venv/bin/activate.csh  # csh or tcsh
  $ . ~/tensorflow/venv/bin/activate.fish      # fish
  ```

**When the Virtualenv is activated, the shell prompt displays as (venv) $.**

**4. Upgrade pip in the virtual environment.**

Within the active virtual environment, upgrade pip:

```shell
(venv)$ pip install -U pip
```
You can install other Python packages within the virtual environment without affecting packages outside the virtualenv.

**5. Install TensorFlow in the virtual environment.**

Choose one of the available TensorFlow packages for installation:

- tensorflow —Current release for CPU
- tensorflow-gpu —Current release with GPU support
- tf-nightly —Nightly build for CPU
- tf-nightly-gpu —Nightly build with GPU support

Within an active Virtualenv environment, use pip to install the package:


```shell
$ pip install -U tensorflow
```

Use pip list to show the packages installed in the virtual environment. Validate the install and test the version:

```shell
(venv)$ python -c "import tensorflow as tf; print(tf.__version__)"
```

