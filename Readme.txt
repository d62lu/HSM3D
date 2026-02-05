1. Please refer to https://github.com/Pointcept/PointTransformerV3 for environment setting.

2. Example data: please download "toronto3d_sp_multi_level5_2" folder from https://drive.google.com/file/d/1G2rCd55epujpdLYM0Xm8c5WsKei-5X2Q/view?usp=sharing

and save it in "data" folder.

3. Pretrain model loading: please download best_model.pth from https://drive.google.com/file/d/1epvhRPofbZAWphlDNJ2R_S-45ySoEnnp/view?usp=drive_link, and save it in "\log\HSM3D\toronto3d\HSM3D\checkpoints" folder.

4. After setting environment,  run:

python train_val_hsp3d.py



Reference: Data format.

Toronto-3D dataset has 4 areas, 3 (L001, l003, l004) for training, 1 (L002) for testing. In the data folder, each area has 5 .npy files with different superpoint clustering coarseness. 

Each npy file of each area has N × D， N means point number (about 4 million), D means feature channels: x, y, z, r, g, b, intensity, geof1, geof2, geof3, geof4, sp_idx, gt_label

[:-2] columns means input features to model, -2 column means superpoint index for each point, -1 column means ground truth label

the superpoint clustering is for point clouds,  i didn't put it in the code because there should be well-designed superpixel methods for 2d image processing. Just keep the data format after your superpixel preprocessing same as the example data.



