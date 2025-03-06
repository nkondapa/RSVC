# uncomment if needed on your machine
# apt-get update && apt-get install nano zip ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ python3-distutils python3-apt -y 

pip install -r requirements.txt

python3 setup.py build_ext --inplace
