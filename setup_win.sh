ENV=env-`basename \`pwd\``
python -m venv $ENV
source $ENV/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt
ipython kernel install --user --name=$ENV --display-name=$ENV

python setup_data.py
