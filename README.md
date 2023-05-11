<img src="./assets/logo.png" width = "330" height = "110" alt="logo" />

<div align="center"><img src="./assets/track.gif" width = "150" height = "150" alt="track" /><img src="./assets/seg.gif" width = "150" height = "150" alt="seg" /><img src="./assets/sil.gif" width = "150" height = "150" alt="sil" /></div>

TrackGait is a sub project of [OpenGait](https://github.com/ShiqiYu/OpenGait) provided by the [Shiqi Yu Group](https://faculty.sustech.edu.cn/yusq/). Implemented a gait recognition system.

## How to use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wZE9wo_Y6Hwgp7tkUbGrCZPAa-ChX2g2?usp=sharing)

## How to run demo

#### Step1. Prepare Environment
```
git clone https://github.com/jdyjjj/TrackGait.git
cd TrackGait
pip install -r requirements.txt
pip install yolox
```
#### Step2. Get checkpoints
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
            └——————human_pp_humansegv2_mobile_192x192_inference_model_with_softmax
```

##### Get the checkpoint of gait model

```
cd TrackGait/OpenGait/demo/checkpoints
mkdir gait_model
cd gait_model
wget https://github.com/ShiqiYu/OpenGait/releases/download/v2.0/pretrained_grew_gaitbase.zip
unzip -j pretrained_grew_gaitbase.zip

```

##### Get the checkpoint of tracking model
```
cd TrackGait/OpenGait/demo/checkpoints/bytetrack_model
pip install --upgrade --no-cache-dir gdown
gdown https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5
```

This is the link of bytetrack, download it and put it in the folder "byte track_model"

- bytetrack_x_mot17 [[google]](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing), [[baidu(code:ic0i)]](https://pan.baidu.com/s/1OJKrcQa_JP9zofC6ZtGBpw)

##### Get the checkpoint of segment model
```
cd TrackGait/OpenGait/demo/checkpoints
mkdir seg_model
cd seg_model
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip
unzip human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip
```

#### Step3. Run demo
```
cd TrackGait/OpenGait
python demo/libs/main.py
```

#### Step4. See the result

```
cd TrackGait/OpenGait/output

output
	 └——————GaitFeatures: This stores the corresponding gait features
   └——————Inputvideos: This is the folder where the input videos are put
   |——————Outputvideos
   |        └——————track_vis
   | 				|				└——————gait_model
	 | 				| 						└——————timestamp
   |        └——————seg_model
   └——————silhouette: This stores the corresponding gait silhouette images
   
   
timestamp: Store the result video of the track here, naming it consistent with the input video. In addition, videos with the suffix "- After. mp4" are obtained after gait recognition.
```

## Authors:

**Open Gait Team (OGT)**

- [Dongyang Jin(金冬阳)](https://faculty.sustech.edu.cn/?p=176498&tagid=yusq&cat=2&iscss=1&snapid=1&go=1&orderby=date), 11911221@mail.sustech.edu.cn
- [Chao Fan (樊超)](https://faculty.sustech.edu.cn/?p=128578&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date), 12131100@mail.sustech.edu.cn
- [Rui Wang(王睿)](https://faculty.sustech.edu.cn/?p=161705&tagid=yusq&cat=2&iscss=1&snapid=1&go=1&orderby=date), 12232385@mail.sustech.edu.cn
- [Chuanfu Shen (沈川福)](https://faculty.sustech.edu.cn/?p=95396&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date), 11950016@mail.sustech.edu.cn
- [Junhao Liang (梁峻豪)](https://faculty.sustech.edu.cn/?p=95401&tagid=yusq&cat=2&iscss=1&snapid=1&orderby=date), 12132342@mail.sustech.edu.cn

## Acknowledgement

- ByteTrack: [bytetrack](https://github.com/ifzhang/ByteTrack)
- Paddleseg: [paddleseg](https://github.com/PaddlePaddle/PaddleSeg)

