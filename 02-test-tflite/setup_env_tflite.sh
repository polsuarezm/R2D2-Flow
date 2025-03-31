me of the virtual environment directory
VENV_DIR="drl_venv"

echo "Creating virtual environment in '$VENV_DIR'..."
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
# For TFLite runtime (lightweight; recommended for edge devices like KV260)
pip install tflite-runtime

# For SPI access from Python
pip install spidev

# Any other packages your DRL code may need
pip install numpy

echo "Setup complete. To activate your venv later, run:"
echo "source $VENV_DIR/bin/activate"
