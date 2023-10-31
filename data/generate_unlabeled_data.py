import os
import imageio
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir',
                    default=None,
                    required=True,
                    help='Path to data directory')

def main(args):
    # Set the path for the source images directory
    src_dir = args.data_dir

    # Set the path for the output directory
    train_path = f"{src_dir}/train_data/"
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path + '/raw')
    else:
        raise Exception("Train folder is non-empty")
    print('Preparing Train Data')

    # Set the patch size and padding
    patch_size = [128, 128, 1]
    pad = [5, 5, 0]

    image_id = 0

    # Loop through each source image in the source directory
    for i in tqdm(range(500)):
        # Read the image
        img_path = os.path.join(src_dir, "real_images", f"{10001 + i}.bmp")
        image = imageio.imread(img_path)

        # Get the dimensions of the source and target image and padding
        p_h, p_w, _ = patch_size
        pad_h, pad_w, _ = pad

        p_h = p_h - 2*pad_h
        p_w = p_w - 2*pad_w

        h, w = image.shape
        patch_num = h // 40
        print(patch_num)
        x_ = np.int32(np.linspace(5, h-5-p_h, patch_num))
        y_ = np.int32(np.linspace(5, w-5-p_w, patch_num))

        grid = np.meshgrid(x_, y_, indexing='ij')

        for i, start in enumerate(list(np.array(grid).reshape(2, -1).T)):
            start = np.array((start[0], start[1], 0))
            end = start + np.array(patch_size)-1 - 2*np.array(pad)

            patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1] +
                        p_w, ], ((pad_h, pad_h), (pad_w, pad_w), ))
            
            if patch.sum() > 10:
                imageio.imwrite(train_path+'raw/sample_'+str(image_id).zfill(6)+'_data.png', patch)
                image_id = image_id+1

    # Print a message indicating that the script has finished running
    print("Unlabeled dataset generation complete.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)