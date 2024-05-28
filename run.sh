sudo apt-get update && 
yes Y | sudo apt install python3-pip && 
cd chat_bot &&
pip install -r requirements.txt &&
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu