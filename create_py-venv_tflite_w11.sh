#needs open cmd... 
#cd C:\Your\Target\Folder

python3 -m venv DRL-tflite-w11
source DRL-tflite-w11/bin/activate
pip install --upgrade pip

pip install -r requirements_venv_tflite.txt

# to check packages installed
# in the virtual environment
pip list   