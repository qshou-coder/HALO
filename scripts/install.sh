sudo apt update
sudo apt install libgl1 git -y

pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation

pip install webdataset imgaug

pip install git+https://github.com/kvablack/dlimp.git

pip install tensorflow_graphics
