# BubblEX
point cloud explainability


starting from

https://github.com/AnTao97/dgcnn.pytorch


## Prediction

python predict.py --exp_name=cls_modelnet --model=dgcnn_cls --test_area=all --dataset=modelnet40 --batch_size=1 --test_batch_size=1 --epochs=500 --test_batch_size=1 --model_path=checkpoints/cls_modelnet/models/model.cls.1024.t7

## Activation and Gradient extraction

python actGradExtract.py

## Visualization Module

### tsne


## Interpretation Module

### gradcam

python explain_gradcam.py

python explain_activation.py

### bubble visualization

https://www.mitsuba-renderer.org/

use of mitsuba0.5 to render


