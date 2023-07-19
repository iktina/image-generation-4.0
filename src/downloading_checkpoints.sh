mkdir checkpoints
cd checkpoints
git clone https://huggingface.co/stabilityai/stable-diffusion-2
cd stable-diffusion-2
echo "stable-diffusion-2"
rm -r -f 768-v-ema.ckpt 768-v-ema.safetensors
cd ..
echo "stable-diffusion-2-1"
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1
cd stable-diffusion-2-1
rm -r -f v2-1_768-ema-pruned.ckpt v2-1_768-ema-pruned.safetensors v2-1_768-nonema-pruned.ckpt v2-1_768-nonema-pruned.safetensors
wget -O DreamShaper_6_BakedVae.safetensors https://civitai.com/api/download/models/109123
wget -O Inkpunk-Diffusion-v2.ckpt https://civitai.com/api/download/models/1138
wget https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/anything-v3-fp16-pruned.safetensors
cd ..
mkdir stable-diffusion-v1-5
cd stable-diffusion-v1-5
wget -O beksinski.ckpt https://huggingface.co/s3nh/beksinski-style-stable-diffusion/resolve/main/model.ckpt
wget https://huggingface.co/Langboat/Guohua-Diffusion/resolve/main/guohua.ckpt
wget https://huggingface.co/KSD2023/mdjrny-v4/resolve/main/mdjrny-v4.ckpt
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
cd ..
wget -O icbinpICantBelieveIts_final.safetensors https://civitai.com/api/download/models/109115
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.yaml
cd ..
mkdir kandinsky
cd kandinsky
git clone https://huggingface.co/kandinsky-community/kandinsky-2-1-prior
git clone https://huggingface.co/kandinsky-community/kandinsky-2-1
git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder
git clone https://huggingface.co/kandinsky-community/kandinsky-2-2-prior

cd ..
git clone --depth 1 --branch main --no-checkout https://github.com/lllyasviel/ControlNet-v1-1-nightly.git
cd ControlNet-v1-1-nightly
git sparse-checkout set annotator
cd ..
mv ControlNet-v1-1-nightly/annotator ./
rm -r -f ControlNet-v1-1-nightly
cd annotator/ckpts
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt

cd ../..

mkdir -p pretrained/vqgan
curl https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/resolve/main/flax_model.msgpack -L --output ./pretrained/vqgan/flax_model.msgpack

# download dalle-mini and dalle-mega
python3 -m wandb login --anonymously
python3 -m wandb artifact get --root=./pretrained/dalle_bart_mini dalle-mini/dalle-mini/mini-1:v0
python3 -m wandb artifact get --root=./pretrained/dalle_bart_mega dalle-mini/dalle-mini/mega-1-fp16:v14 