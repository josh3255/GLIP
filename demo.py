import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from io import BytesIO
from PIL import Image

import cv2
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from argparse import ArgumentParser

image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']
video_extensions = ['mp4', 'avi', 'wmv', 'mov', 'flv', 'mkv', 'mpeg', 'webm']

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--source', help='image/video path')
    parser.add_argument('--caption', help='caption to detect')
    parser.add_argument('--config', \
            default='configs/pretrain/glip_Swin_T_O365_GoldG.yaml', help='config file path')
    parser.add_argument('--weight', \
            default='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth', help='weight file path')

    args = parser.parse_args()
    return args

def image_load(path : str):
    img = Image.open(path).convert('RGB')
    img = np.array(img)[:, :, [2, 1, 0]]
    return img

def imsave(img, caption):
    plt.imsave('demo/demo.jpg', img[:, :, [2, 1, 0]])

def demo(args):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(["MODEL.WEIGHT", args.weight])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    
    if args.source.split('.')[-1] in image_extensions:
        img = image_load(args.source)
        result, _ = glip_demo.run_on_web_image(img, args.caption, 0.5)
        imsave(result, args.caption)

    elif args.source.split('.')[-1] in video_extensions:
        cap = cv2.VideoCapture(args.source)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_path = './demo/{}'.format(args.source.split('/')[-1])
        out = cv2.VideoWriter(vid_path, fourcc ,30, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            img = Image.fromarray(frame).convert('RGB')
            img = np.array(img)[:, :, [2, 1, 0]]

            result, _ = glip_demo.run_on_web_image(img, args.caption, 0.5)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            out.write(result)
        out.release()
    

def main():
    args = parse_args()
    demo(args)

if __name__ == "__main__":
	main()