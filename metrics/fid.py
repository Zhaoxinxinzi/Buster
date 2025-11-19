from cleanfid import fid
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

score = fid.compute_fid('img1_folder', 'img2_folder',  mode="clean")
print(f'FID Score: COCO10000 {score}')


    
