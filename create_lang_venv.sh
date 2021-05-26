#!/usr/bin/env bash

VENVNAME=langvenv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install transformers
pip install seaborn

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

python -m spacy download en_core_web_sm

deactivate
echo "build $VENVNAME"