
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
demo
   |——————checkpoints
   |        └——————bytetrack_model
   |        └——————gait_model
   |        └——————seg_model
   └——————libs
   └——————output



checkpoints
   |——————bytetrack_model
   |        └——————bytetrack_x_mot17.pth.tar
   |        └——————yolox_x_mix_det.py
   |
   └——————gait_model
   |        └——————xxxx.pt
   └——————seg_model
            └——————xxx
```

```
cd /OpenGait/demo/checkpoints
mkdir gait_model
cd gait_model
wget https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_grew_model.zip
unzip -j pretrained_grew_model.zip

```

##### prepare track checkpoint
```
cd /OpenGait/demo/checkpoints/bytetack_model
pip install --upgrade --no-cache-dir gdown
gdown https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5
```

- bytetrack_x_mot17 [[google]](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing), [[baidu(code:ic0i)]](https://pan.baidu.com/s/1OJKrcQa_JP9zofC6ZtGBpw)

##### prepare seg checkpoint
```
cd /OpenGait/demo/checkpoints
mkdir seg_model
cd seg_model
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model_with_softmax.zip
unzip human_pp_humansegv2_lite_192x192_inference_model_with_softmax.zip
```

#### Step3. Run demo
```
cd /OpenGait/
python demo/libs/main.py
```

