#  🖼️ Adaptive Convolutions for Structure-Aware Style Transfer

A pytorch implementation of the [Adaptive Convolutions for Structure-Aware Style Transfer](https://studios.disneyresearch.com/app/uploads/2021/04/Adaptive-Convolutions-for-Structure-Aware-Style-Transfer.pdf)

An Encoder-Decoder based style-transfer model with **`AdaConv`** at it's core to allow for the simultaneous transfer of both statistical and structural information from style images to content images.

<p align="center">
   <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/d00bfacebef0cc5abae0cff1c552664a30179648/4-Figure2-1.png" alt="AdaConv architecture"/>
</p>

## Dataset

Used [cocodataset](https://cocodataset.org/#download) for content images and [Painter by Numbers](https://www.kaggle.com/competitions/painter-by-numbers/data) for style images.

Download, extract and move the train and test folders from content and style dataset into the folder tree strucuture as shown below 

	data/
	 └─ raw/
         ├─ content/
         │   ├─ train2017/
         │   │   ├─ 0.jpg
         │   │   ├─ ...
         │   │   └─ N.jpg
         │   └─ test2017/
         │       ├─ 0.jpg
         │       ├─ ...
         │       └─ N.jpg
         └─ stytle/
             ├─ train/
             │   ├─ 0.jpg
             │   ├─ ...
             │   └─ N.jpg
             └─ test/
                 ├─ 0.jpg
                 ├─ ...
                 └─ N.jpg

## Training

Create a `config.yaml` file with field names specified in the `hyperparam.py`

> **RUN**

	python train.py -c <path to config.yaml> -d <path to dataset> -l <log directory path>

Monitor the training using tensorboard with the tensorboard log file under the *`tensorboard`* folder under the log directory 

Once the training is done, the checkpoints can be found in the *`ckpts`* folder in log directory

    logdir/
     ├─ config.yaml
     ├─ ckpts/
     │	 ├─ last.pt
     │	 ├─ model_step_*.pt
     │	 ├─ model_step_*.pt
     │	 └─ model_step_*.pt
     └─ tensorboard/
         └─ {%Y%m%d-%H%M%S}
     	     └─ events.out.tfevents.*

## Testing

Basic inference testing on content and style image to create a matrix grid images of styles (rows) x contents (columns)

> **RUN**

	python test.py --config <trained config.yaml> --model_ckpt <model ckpt file path> --content_path <content image(s) path/dir> --style_path <style image(s) path/dir> --output_path <output image file path>


## References

- [Adaptive Convolutions for Structure-Aware Style Transfer](https://openaccess.thecvf.com/content/CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.pdf)
- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868)