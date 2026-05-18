# Flow Matching for Applied Mathematicians, tutorial code
## Author: Alasdair Newson

This code implements the Flow Matching algorithm for data generation, presented in the paper "An Introduction to Flow Matching for Applied Mathematicians". It implements the method specifically for **images**.

## Main function arguments.

The main function is named ```main_flow_matching.py``` and allows several arguments:
- ```target_dataset``` (string): name or directory containing target distribution. Must be a recognised string (mnist, cifar10) or a valid directory with images (png, jpg or tiff) inside.
- ```source_dataset``` (string): name or directory containing target distribution. Must be a recognised string (gaussian) or a valid directory with images (png, jpg or tiff) inside.
- ```shuffle_target``` (int): shuffle (1) or no shuffle (0) of target distribution
- ```shuffle_source``` (int): shuffle (1) or no shuffle (0) of source distribution
- ```train``` (int): train (1) or test (0). If test, you must put the path to the saved model checkpoint. This checkpoint must be coherent with the specified architecture (see ```arch```)
- ```model_path``` (str): path of the saved model checkpoint. Default=""
- ```arch``` (string): name of the architecture to use (default: 'unet_tiny'). Options : unet_tiny, unet_small, unet
- ```save_dir``` (string): directory to save results to
- ```epochs``` (int): number of epochs to train model (default: 10)
- ```batch_size``` (int): number of samples in a batch (default: 16)
- ```image_size``` (int): size of the image (default: -1). If ``target_dataset`` is in the list of known databases, then the image size is already fixed, and this parameter is ignored. Otherwise (custom dataset)
    - If the image size is equal to -1, then the behaviour depends on the ``target_dataset``.  If it is not known (custom database), then the image size is fixed to the size of the first image of the ```target_dataset``` directory
    - If the image size is greater than 0, then we consider that the image is square, and equal to size ``image_size``$\times$``image_size``.
- ```n_channels``` (int): the number of channels of the image (default: 1) and the noise (these are necessarily the same).
- ```steps_sampler```(int): number of steps in the Euler scheme (default: 40).
- ```schedule``` (string): name of schedule to use in Flow Matching (default: 'linear').
- ```sample_every``` (int): regular interval of epochs after which the model's generation results are sampled (default: 20)
- ```samples_n``` (int): number of images to sample during generation
- ```lr``` (float): learning rate for the ```torch.optim.AdamW``` optimiser(default: 3e-4)
- ```ckpt_every``` (int):  regular interval of epochs after which the model is saved to the 'save_dir' directory (default: 20)
- ```wd``` (float): weight decay (default: 0)
- ```seed``` (int): random seed for reproducibility

### Example usage 

python main_flow_matching.py \
    --target_dataset "mnist" \
    --source_dataset "gaussian" \
    --save_dir ./results \
    --epochs 100 \
    --batch_size 64 \
    --steps_sampler 40 \
    --arch "unet_tiny" \
    --schedule "linear" \
    --sample_every 1 \
    --samples_n 4\
    --lr 3e-4\
    --ckpt_every 40

## Source and target distributions

The code can handle various different types of distributions for target and source data. Both the target and source data can be of the following types :
- Images in a directory. All the images must be in the same folder (no sub-folders). The following formats are accepted: ".png", ".jpg", ".jpeg", ".bmp", ".tiff". The image size is automatically ascertained, and by default the reference image size is that of the target. In this situation, the source images are resized to conform to the target image size.
- Predefined databases : mnist, cifar10. The code will download them locally.
- Predefined random distributions : gaussian

## Image size and number of channels

If you specify a custom database, in the ```data_root``` folder, you can either specify the image size and number of channels, or let the code determine them automatically (by setting the ```image_size``` and ```n_channels``` to -1). If you choose the first option, then the code will consider that the image size is square, and resize all images to this square size. If you choose the second option, the code will use the first image it finds in the ```data_root``` directory and use its size and number of channels for all images. Non-square image sizes are allowed in this case.

If both the target and source are random distributions (only Gaussians are handled at this time), then the image size must be given by the user.

## Testing

The code may be used in testing, with a pretrained model: set ```--train 0``` in this case. In this situation, the square image size must be provided by the user, since we cannot ascertain it from the model itself (only convolutional architectures are currently handled).


## Results

The output of the algorithm will be written to the ```save_dir``` directory (default "./results"). Furthermore, the code will create a results sub-directory corresponding to the name of the target distribution. If the name is not specified, the code will use the name of the directory in which the training data is found.