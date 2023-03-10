# GPD Classification 
__Gastric Precancerous Diseases__ classification
using __YOLOv5__ with `yolov5x` model and a customized model (`GPD-Model`).
## Dataset description
This dataset has _three_ classes which splited into **train**, **test**, and **validation** subsets. 

It is described in tabel below.

|dataset   |   Erosion |   Polyp |   Ulcer |
|:-----|:---------:|:-------:|:-------:|
|train |       889 |     868 |     891 |
|test  |       176 |     189 |     186 |
|valid |       144 |     159 |     165 |


You can find the actuale images with their corresponding labels under each subset directory.

Notice:
>In this case, the bounding box has drawn over the whole image.

## GPD-Model

The previous attempts for only classification objective.

You can find the prediction COLAB Notebook on this [link ](https://colab.research.google.com/drive/1teZKBQfhxQSyE1zrAW_MObsyppXuWblW?usp=sharing).

## YOLOv5 Model

We train a model by using `yolov5x` architecture from scratch (there was no initial weights) and leave the default configuration intact. We clone Yolov5 implementation from this repository [github](https://github.com/ultralytics/yolov5).

Here is our training arguments:

```bash
python train.py --img 32 --batch 1024 --epochs 100 --patience 100 --cfg yolov5x.yaml --data data.yaml --weights ' '
```

>You can also find our exact configuration from `yolov5-hyperparameters.yaml`.

### Output model

The original output model from our best training results `best.pt`, which you can download it from [here](https://drive.google.com/file/d/10jxDnx17iyr6eJtSPFmXLBAUAy9TVoEJ/view?usp=share_link).

Then we exported our original model to nvidia optimized version by TensorRT in order to reduce time consumption for doing inferences. And our optimized model version is `best.engine`. here is the [link](https://drive.google.com/file/d/1-Auz91Q95tJpuZK4QxzjGb4YyI6wKKVa/view?usp=share_link) to download. 

## Cantact information 
For further information .

gmail : sina.behnam.ai@gmail.com & r_shamsaee@hotmail.com & shokoufeh.hatami1994@gmail.com
