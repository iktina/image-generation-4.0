import time 
import json
import torch
import logging
import argparse
import pandas as pd
from concurrent import futures
from modules.models.Loading import ModelLoading
from modules.generation.generation import Generation

import grpc
import Generation_pb2
import Generation_pb2_grpc

torch.set_grad_enabled(False)
dtype = "float16"

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

parser = argparse.ArgumentParser(description='')
parser.add_argument("--host", type=str, default="0.0.0.0",  help= "host")
parser.add_argument("--port", type=int, default=8010,  help= "port" )

args = parser.parse_args()

class image_generationServicer(Generation_pb2_grpc.image_generationServicer):
    def __init__(self):
        
        self.current_model = 'SD2_1'
        loading = ModelLoading()
        model_dalle = loading.MinDalleLoad('cuda:1')
        self.df_models = pd.read_csv("model_paths.csv")
        
        self.generation = Generation(model_dalle)
        self.generation.formPipe("checkpoints/stable-diffusion-2-1", self.current_model)
        
    def Gen(self, request, context):
        
        path, add_prompt, request_model, sizeH, sizeW = self.generation.getPath(request, self.df_models)
            
        if self.current_model != request_model and request_model != "dalle":
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            self.generation.formPipe(path, request_model)

        images_list = self.generation.Gen(request, request_model, add_prompt, sizeH, sizeW)
        self.current_model = request_model
        
        dictionary = {
        "images": [byt.decode('ascii') for byt in images_list]
        }
        
        json_object = json.dumps(dictionary, indent=4)
        
        return Generation_pb2.Image(image1=json_object)
        
        
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    Generation_pb2_grpc.add_image_generationServicer_to_server(image_generationServicer(), server)
    server.add_insecure_port('{}:{}'.format(args.host, args.port))
    server.start()
    print('Server start')

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    serve()
