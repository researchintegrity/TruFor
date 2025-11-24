import sys
import os
import argparse
import time
import signal
import logging
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

# Add current directory to path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Also add parent directory if needed (as in original script)
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from config import update_config
from config import _C as config
from data_core import myDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    print("[STATUS] TIMEOUT", flush=True)
    logger.error("Timeout reached. Exiting.")
    sys.exit(1)

def save_visualization(rgb_path, pred_map, conf_map, output_path):
    """
    Save a visualization of the results.
    Generates a composite image with Original, Localization Map, and Confidence Map.
    """
    try:
        # Load original image
        img = Image.open(rgb_path).convert('RGB')
        
        # Create figure
        # Adjust figsize as needed
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original Image
        axs[0].imshow(img)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        
        # Localization Map
        # Using RdBu_r colormap as in visualize.py
        # We resize the map to match the image for better display if needed, 
        # but imshow handles different sizes in subplots fine.
        axs[1].imshow(pred_map, cmap='RdBu_r', vmin=0, vmax=1)
        axs[1].set_title('Localization Map')
        axs[1].axis('off')
        
        # Confidence Map
        axs[2].imshow(conf_map, cmap='gray', vmin=0, vmax=1)
        axs[2].set_title('Confidence Map')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return True
    except Exception as e:
        logger.error(f"Error saving visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run TruFor Detection')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
    parser.add_argument('-in', '--input', type=str, required=True, help='input image path')
    parser.add_argument('-out', '--output', type=str, default='../output', help='output folder')
    parser.add_argument('--timeout', type=int, default=0, help='timeout in seconds')
    parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Set timeout if specified
    if args.timeout > 0:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(args.timeout)
    
    print("[STATUS] INITIALIZING", flush=True)
    
    # Update config
    update_config(config, args)
    
    device = 'cuda:%d' % args.gpu if args.gpu >= 0 else 'cpu'
    
    # Setup input
    input_path = args.input
    output_dir = args.output
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load Model
    print("[STATUS] LOADING_MODEL", flush=True)
    
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        # Fallback or error
        logger.error("Model file not specified in config")
        sys.exit(1)
        
    try:
        checkpoint = torch.load(model_state_file, map_location=torch.device(device))
        
        if config.MODEL.NAME == 'detconfcmx':
            from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
            model = confcmx(cfg=config)
        else:
            raise NotImplementedError(f'Model {config.MODEL.NAME} not implemented')
            
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Process Image
    print("[STATUS] PROCESSING_IMAGE", flush=True)
    
    try:
        # We use myDataset to handle loading and preprocessing
        # It expects a list of images
        test_dataset = myDataset(list_img=[input_path])
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        
        with torch.no_grad():
            for index, (rgb, path) in enumerate(testloader):
                rgb = rgb.to(device)
                
                # Inference
                # model returns: pred, conf, det, npp
                pred, conf, det, npp = model(rgb)
                
                # Process outputs
                if conf is not None:
                    conf = torch.squeeze(conf, 0)
                    conf = torch.sigmoid(conf)[0]
                    conf = conf.cpu().numpy()

                pred = torch.squeeze(pred, 0)
                pred = F.softmax(pred, dim=0)[1]
                pred = pred.cpu().numpy()
                
                # Generate Output Filename
                filename = os.path.basename(input_path)
                basename = os.path.splitext(filename)[0]
                output_filename = f"{basename}_trufor_result.png"
                output_path = os.path.join(output_dir, output_filename)
                
                print("[STATUS] SAVING_RESULTS", flush=True)
                
                # Save Visualization
                if save_visualization(input_path, pred, conf, output_path):
                    print(f"[STATUS] COMPLETED {output_filename}", flush=True)
                else:
                    print("[STATUS] FAILED_VISUALIZATION", flush=True)
                    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
