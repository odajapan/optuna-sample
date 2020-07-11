ENV=env-`basename \`pwd\``
python3 -m venv $ENV
source $ENV/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
ipython kernel install --user --name=$ENV --display-name=$ENV

python setup_data.py
