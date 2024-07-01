# Intro to CPS Security - Mini Project 4

## Environment Setup
To setup the development environment, you need to have `python3` installed in your machine. Then, clone this github repo in your machine, `cd` into it, and run:

```
python3 -m venv .venv
```

This will setup a virtual environment in your local repo so you can install the dependencies. Activate the virtual environment in the repo with:

In Windows:
``` 
.\.venv\Scripts\activate
```

In MacOS:
```
source .venv/bin/activate
```

With the virtual environment activated, install the dependencies listed in your requirements.txt file by running:

```
pip3 install -r requirements.txt
```

If you need jupyter notebook, first run this command to create a kernel based on the virtual environment and then use the next command to launch an instance of it.
```
python3 -m ipykernel install --user --name=mp4env --display-name="mp4env"
jupyter notebook
```

## Running Training and Test Workflows
Once you completed the code skeleton, please navigate to `mp4-machine-learning-template` directory in terminal. Then you can run the training and testing workflows with:

```
python3 src/train.py

python3 src/test.py
```
