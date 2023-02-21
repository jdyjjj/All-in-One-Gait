
Thanks for [bytetrack](https://github.com/ifzhang/ByteTrack) and [paddleseg](https://github.com/PaddlePaddle/PaddleSeg).


## Demo Links
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19EVnzwaCpu6RzsI90GJuPLePvGALAQPr?usp=sharing)


How to run demo
------------------------------------------
#### Step1. Prepare Environment
```
cd OpenGait
pip install -r requirements.txt
```
#### Step2. Prepare checkpoints
##### prepare gait checkpoint
```
cd /OpenGait/demo/checkpoints
mkdir gait_model
cd gait_model
wget https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_grew_model.zip
unzip -j pretrained_grew_model.zip
```

##### prepare track checkpoint
```
cd /OpenGait/demo/checkpoints
mkdir bytetack_model
cd bytetack_model
pip install --upgrade --no-cache-dir gdown
gdown --id "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5"
```


##### prepare seg checkpoint
```
cd /OpenGait/demo/checkpoints
mkdir seg_model
cd seg_model
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip
```

#### Step3. Run demo
```
cd /OpenGait/
python libs/demo/main.py
```

