# Vidy NLP Protocol
Python implementation of the NLP models used by the contextual ad placement platform

<p align="center"><img width="70%" src="images/vidy-nlp-protocol.png" /></p>

Vidy has invented the first single-page invisible embed layer for video, run on the Ethereum blockchain.
With just a hold, users can now reveal tiny hyper-relevant videos hidden behind the text of any page on the web, unlocking a whole new dimension to the internet.

## Models
- [Category Detection](#category-detection) 
- [Emoji Analysis](#emoji-analysis)
- [Offensive Content](#offensive-content)
- [Sentiment Analysis](#sentiment-analysis)
- [Keyword Extraction](#keyword-extraction)
- [Related Keywords](#related-keywords)

## Category Detection

  Given a sentence, our model is able to classify it amongst the following categories:

  ```
  arts, automobiles, books, business, corrections, education, fashion, health, insider, jobs, magazine, movies, multimedia, nyregion, obituaries, opinion, politics, reader-center, realestate, science, smarter-living, sports, t-magazine, technology, theater, travel, upshot, us, well, world
  ```

  The model was trained on a massive collection of news articles.

### Architecture
  * convolution banks + highway networks + fc
  * used pretrained word embeddings
  * vocabulary: 50k
  * inputs: last 100 words in summary.
  * targets: category.

### Evaluation on dev set
  * Accuracy at 1: 67%
  * Accuracy at 3: 90%

## Offensive Content

  The goal is to determine if a text is sexual or not.
  Trained on news articles and websites with sexual content.

### Architecture
  * vocabulary: 400k from glove.100d
  * inputs: last 100 words in text.
  * Convolution Banks
  * Hiway Networks
  * GRU -> Last hidden vector
  * FC to 2 labels (sexual vs. clean)

### Results
  * global step: 23088
  * precision: 272318/273260=1.00 (sexual out of sexual)
  * recall: 272318/273119=1.00 (correctly predicted ones out of true sexual samples)
  * acc: 544689/546432=1.00

## Emoji Analysis

  Mutli-class multi-label classification of a sentence's emotion.

### Architecture
  * vocabulary: 400k from glove.100d
  * inputs: last 100 words in text.
  * Convolution Banks
  * Hyway Networks
  * GRU -> Last hidden vector
  * FC to 1.4K (Emojis)

## Keyword Extraction

  Determine which words in a sentence are the most relevant keywords.

### Architecture
  * Binary Classification (Sequence labelling)
  * vocabulary: 400k from glove.100d
  * inputs: last 100 words in text
  * Two residual bidirectional GRUs
  * FC to 2 labels (is_keyword vs. not_keyword)

## Sentiment Analysis

  Determine if the emotion of a text is sad or not.
  Binary Classification of tweets associated with emojis.

### Architecture
  * vocabulary: 400k from glove.100d
  * inputs: last 100 words in text.
  * Convolution Banks
  * Hyway Networks
  * GRU -> Last hidden vector
  * FC to 2 labels (sad vs. not sad)


## Related Keywords

  Given a keyword, list related words.

  Through word embedding a vocabulary is learned to map words or phrases to vectors of real numbers.

  The underlying idea is that "a word is characterized by the company it keeps".

  Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension.

## Getting Started
- [Installing TensorFlow](#installing-tensorflow) 
- [Downloading models](#downloading-pretrained-models)
- [Setting up Flask](#setting-up-flask)
- [Training](#training)
- [Launching the services](#launching-the-services)
- [Calling the apis](#calling-the-apis)

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


## Downloading Pretrained Models

 We offer pretrained versions of our models for download.

 On the [training](#training) section you can read more about training our models from scratch on your own dataset.

 Download all our models on [http://storage.googleapis.com/vidy-nlp/data.tar](http://storage.googleapis.com/vidy-nlp/data.tar)

## Setting up Flask

<p align="center"><img width="70%" src="images/flask.png" /></p>


Flask is a lightweight [WSGI](https://wsgi.readthedocs.io/) web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications. It began as a simple wrapper around [Werkzeug](https://www.palletsprojects.com/p/werkzeug/) and [Jinja](https://www.palletsprojects.com/p/jinja/) and has become one of the most popular Python web application frameworks.

### Installation
 Install and update using pip:

  ```shell
  $ pip install -U Flask
  ```

### Checking installation

 Create a hello.py file and insert the following:

 ```
 from flask import Flask

 app = Flask(__name__)

 @app.route('/')
 def hello():
     return 'Hello, World!'
 ```
 
 Then launch the test server with

 ```
 $ env FLASK_APP=hello.py flask run
 * Serving Flask app "hello"
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 ```

 Call the main endpoint. Your should receive the message: Hello, World!

 ```
 curl http://127.0.0.1:5000
 ```

## Training

W.I.P

## Launching the services

```
export FLASK_APP=category_service.py
flask run --host=0.0.0.0 --port=5000
```

```
export FLASK_APP=emoji_service.py
flask run --host=0.0.0.0 --port=5001
```

```
export FLASK_APP=offensive_service.py
flask run --host=0.0.0.0 --port=5002
```

```
export FLASK_APP=sentiment_service.py
flask run --host=0.0.0.0 --port=5003
```

```
export FLASK_APP=keywords_service.py
flask run --host=0.0.0.0 --port=5004
```

```
export FLASK_APP=related_service.py
flask run --host=0.0.0.0 --port=5005
```


## Calling the apis

### Category Detection

```
curl -X POST -d '[{"text":"The Oakland A’s Are Crashing the Playoff Race Again"},{"text":"I Used Apple’s New Controls to Limit a Teenager’s iPhone Time (and It Worked!)"}]' -H 'Content-type: application/json' http://130.211.155.193:5000/analyze-category
```
```
[  
   {  
      "score":-0.17566493153572083,
      "category":"sports"
   },
   {  
      "score":-1.8768072128295898,
      "category":"technology"
   }
]
```

### Emoji Analysis

```
curl -X POST -d '[{"text":"A Patriotic Fourth: What Does That Mean Now?"},{"text":"Five Times the Internet Was Actually Fun in 2017"}]' -H 'Content-type: application/json' http://130.211.155.193:5001/analyze-emoji
```
```
[  
   [  
      "Flag of United States"
   ],
   [  
      "Face with tears of joy"
   ]
]
```

### Offensive Content

```
curl -X POST -d '[{"text":"Her ex-boyfriend had sex with her best friend."},{"text":"Trump Says Fed, China and Europe Hurt U.S. Economy"}]' -H 'Content-type: application/json' http://130.211.155.193:5002/analyze-offensive
```
```
[
   0,
   1
]
```

### Sentiment Analysis

```
curl -X POST -d '[{"text":"At this time I must confirm my exit from a show I&#39;ve called home for 3 years, with what is the most talented ensemble on television today, the ABC star said in a statement"},{"text":"Analysts say there is still room for diplomacy \u2014 but with more realistic goals."}]' -H 'Content-type: application/json' http://130.211.155.193:5003/analyze-sentiment
```
```
[  
   0,
   1
]
```

### Keyword Extraction

```
curl -X POST -d '[{"text":"Global warming could wipe out most of the country’s remaining cedar forests by the end of the century."}]' -H 'Content-type: application/json' http://130.211.155.193:5004/analyze-keywords
```
```
[  
   [  
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1
   ]
]
```

### Related Keywords

```
curl -X POST -d '{"word":"car"}' -H 'Content-type: application/json' http://130.211.155.193:5005/analyze-related
```
```
[  
   {  
      "word":"cars",
      "distance":4.631352834271344
   },
   {  
      "word":"vehicle",
      "distance":4.692127319987073
   },
   {  
      "word":"truck",
      "distance":4.970341072520736
   },
   {  
      "word":"driver",
      "distance":5.188564331848413
   },
   {  
      "word":"driving",
      "distance":5.5373158679388235
   },
   {  
      "word":"vehicles",
      "distance":6.009785233931166
   },
   {  
      "word":"automobile",
      "distance":6.013379499825719
   },
   {  
      "word":"drove",
      "distance":6.125938972284942
   },
   {  
      "word":"parked",
      "distance":6.210490835926898
   },
   {  
      "word":"motorcycle",
      "distance":6.223316688091225
   }
]
```
