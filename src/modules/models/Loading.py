import torch
from min_dalle import MinDalle
from omegaconf import OmegaConf

dtype = "float16"

class ModelLoading():
    def __init__(self):
       pass
    
    def MinDalleLoad(self, device):
        self.model_dalle = MinDalle(
                    dtype=getattr(torch, dtype),
                    device=device,
                    is_mega=True, 
                    is_reusable=True,
                    models_root='pretrained')
        
        return self.model_dalle
    
    
    