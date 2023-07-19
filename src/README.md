# For environment

conda env create -f env.yml

conda activate SD

sh installer.sh

# To download models

sh downloading_checkpoints.sh

# For start

`Server`
```python
python main.py
```

`Client`
```python
python Client.py --model <model name> --prompt <your prompt> --negative_prompt <your negative prompt>
```

Model names:
* SD2_1
* SD2
* SD1_5
* guohua
* waifu
* inkpunk
* dream_shaper
* open_journey
* icbinp
* dalle 
* scribble
* img2img
* kandinsky2_1
* kandinsky2_2
