import io
import time
import json
import base64
import logging
import argparse
from PIL import Image

import grpc
import Generation_pb2
import Generation_pb2_grpc

parser = argparse.ArgumentParser()

parser.add_argument("--host", default="127.0.0.1", type=str, help="host")
parser.add_argument("--port", default=8010, type=int, help="port")

parser.add_argument("--seed", default=45, type=int, help="seed")
parser.add_argument("--n_images", default=3, type=int, help="number of images")
parser.add_argument("--steps", default=50, type=int, help="steps")
parser.add_argument("--img", default=None, type=str, help="path to image")
parser.add_argument("--H", default=768, type=int, help="height")
parser.add_argument("--W", default=768, type=int, help="width")
parser.add_argument("--guidance_scale", default=7.5, type=float, help="guidance_scale")

parser.add_argument("--prompt_style", default=' ', type=str, help="add prompt")
parser.add_argument("--model", default='SD2_1', type=str, help="add prompt")

parser.add_argument("--prompt", default="cabela’s tent futuristic pop up family pod, cabin, modular, person in foreground, mountainous forested wilderness open fields, beautiful views, painterly concept art, joanna gaines, environmental concept art, farmhouse, magnolia, concept art illustration by ross tran, by james gurney, by craig mullins, by greg rutkowski trending on artstation", type=str, help="text for generation")
parser.add_argument("--negative_prompt", default="", type=str, help="text for generation")
args = parser.parse_args()

def image_generation(stub):
    s_time = time.time()
    
    if args.img != None:
        with open(args.img, 'rb') as f:
            img_bytes = f.read()
    else:
        img_bytes = None
    
    result = stub.Gen(Generation_pb2.Text(model=args.model,
                                          sentence=args.prompt,
                                          seedVal=args.seed,
                                          n_images=args.n_images,
                                          steps=args.steps,
                                          prompt_style=args.prompt_style,
                                          image=img_bytes, 
                                          negative_prompt=args.negative_prompt,
                                          H=args.H,
                                          W=args.W,
                                          guidance_scale=args.guidance_scale))
    r_time = time.time() - s_time

    print('\n########################################################################################\n')
    print("{:.3}s\n{}".format(r_time, type(result.image1)))
    print('\n########################################################################################\n')
    
    # Check Save
    res = json.loads(result.image1)
    idx = 0
    for image in res['images']:
        image_byte = Image.open(io.BytesIO(base64.b64decode(image)))
        image_byte.save(f"output/result{idx}.png", "PNG")
        idx += 1

def run():
    options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)] 
    with grpc.insecure_channel('{}:{}'.format(args.host, args.port), options=options) as channel:
        stub = Generation_pb2_grpc.image_generationStub(channel)
        image_generation(stub)

if __name__ == '__main__':
  logging.basicConfig()
  run()