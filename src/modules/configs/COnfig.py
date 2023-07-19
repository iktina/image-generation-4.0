import argparse

class COnfig():
      def __init__(self):
        
        self.outdir = '/outputs_SD/txt2img-samples'
        self.ddim_steps = 100
        self.plms = True
        self.laion400m = False
        self.seed = 42
        self.config = 'SD/configs/stable-diffusion/v1-inference.yaml'
        self.ckpt = 'SD/models/ldm/stable-diffusion-v1-4/sd-v1-3.ckpt'
        self.precision = 'autocast'
        self.n_rows = 1
        self.fixed_code = False
        self.from_file = False
        self.C = 4
        self.H = 352
        self.W = 352
        self.f = 8
        self.n_samples = 1
        self.n_iter = 1
        self.scale = 7.5
        self.ddim_eta = 0.0
        self.skip_save = False
        self.skip_grid = False
        self.torchscript = False
        self.ipex = False
        self.device = 'cuda'
        self.steps = 50
             
        
class SDConf():
    
    def __init__(self):
        pass
  
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a bear with baloons",
            help="the prompt to render"
        )
        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs/txt2img-samples"
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
            default=False,
        )
        parser.add_argument(
            "--dpm",
            action='store_true',
            help="use DPM (2) sampler",
            default=False,
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across all samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=768,
            help="image height, in pixel space",
        )
        parser.add_argument(
            "--W",
            type=int,
            default=768,
            help="image width, in pixel space",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=4,
            help="latent channels",
        )
        parser.add_argument(
            "--f",
            type=int,
            default=8,
            help="downsampling factor, most often 8 or 16",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=1,
            help="how many samples to produce for each given prompt. A.k.a batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=1,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=9.0,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file, separated by newlines",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="configs/stable-diffusion/v2-inference-v.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="checkpoints/v2-1_768-ema-pruned.ckpt",
            help="path to checkpoint of model",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=47,
            help="the seed (for reproducible sampling)",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default="autocast"
        )
        parser.add_argument(
            "--repeat",
            type=int,
            default=1,
            help="repeat each prompt in file this often",
        )
        parser.add_argument(
            "--device",
            type=str,
            help="Device on which Stable Diffusion will be run",
            choices=["cpu", "cuda"],
            default="cuda"
        )
        parser.add_argument(
            "--torchscript",
            action='store_true',
            help="Use TorchScript",
            default=False,
        )
        parser.add_argument(
            "--ipex",
            action='store_true',
            help="Use Intel® Extension for PyTorch*",
            #default=False,
        )
        parser.add_argument(
            "--bf16",
            action='store_true',
            help="Use bfloat16",
            #default=False,
        )
        opt = parser.parse_args()
        return opt