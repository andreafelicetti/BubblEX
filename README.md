# BubblEX
point cloud explainability


## Prediction

python predict.py --exp_name=cls_modelnet --model=dgcnn_cls --test_area=all --dataset=modelnet40 --batch_size=1 --test_batch_size=1 --epochs=500 --test_batch_size=1 --model_path=checkpoints/cls_modelnet/models/model.cls.1024.t7

## Activation and Gradient extraction

python actGradExtract.py

## tsne


## gradcam

python explain_gradcam.py

python explain_activation.py

### bubble visualization



use of mitsuba0.5 to render


