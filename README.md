# OBMO: One Bounding Box Multiple Objects for Monocular 3D Object Detection

This project is based on our paper: **OBMO: One Bounding Box Multiple Objects for Monocular 3D Object Detection** and the monocular 3D detector: **Geometry Uncertainty Projection Network for Monocular 3D Object Detection**. 

## Usage

The code will be available soon. We currently only provide two checkpoints: [val](https://drive.google.com/file/d/13u4eUhy4StwhLd9IQU56fCRzV9a7D8xN/view?usp=sharing) and [test](https://drive.google.com/file/d/13P7NkqrmTh49deMvBHNP2jjnsw85yu6F/view?usp=sharing), you can test directly with the code of [GUPNet](https://github.com/SuperMHP/GUPNet).

## Offline version

We offer [an offline version](tools/offline_OBMO.py) to quickly test whether OBMO module benefits your model. 

``` sh
python tools/offline_OBMO.py [pred] [calib]
```

We tested the offline version on models such as PatchNet, and the performance were greatly improved. 

Note that the offline results do not necessarily represent the results after training with the OBMO module.

## Acknowlegment

This code benefits from the excellent works: [GUPNet](https://github.com/SuperMHP/GUPNet).
