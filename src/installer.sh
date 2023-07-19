# conda env create -f env.yml
# conda activate SD

pip install min_dalle 

conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch 
pip install transformers invisible-watermark 
pip install -e . 
cd .. 
export CUDA_HOME=/usr/local/cuda-11.4 
conda install -c nvidia/label/cuda-11.4.0 cuda-nvcc 
conda install -c conda-forge gcc 
conda install -c conda-forge gxx_linux-64==9.5.0 
git clone https://github.com/facebookresearch/xformers.git 
cd xformers 
git submodule update --init --recursive 
pip install -r requirements.txt 
pip install triton
pip install -e . 
cd ../stablediffusion 

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers 
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .

pip install accelerate
pip install diffusers==0.12.1
pip install --upgrade diffusers[torch]

