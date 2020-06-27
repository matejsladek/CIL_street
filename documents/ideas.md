# TODO:

- pretraining with GMaps/SpaceNet (fine tuning on the original dataset with lower learning rate or frozen encoder)
- add Jonathan's images to dataset
- ~~check values for image augmentations: brightness, color shift~~
- ~~try different color space representations for input~~
- finish K-NN post processing
- ~~different MTL architecture (?)~~
- GAN based post processing
- stacking/voting (?)

# SIMILAR PROJECTS
- [DeepGlobe: huge competition with many solutions](http://deepglobe.org/leaderboard.html)
    - [paper](https://www.researchgate.net/publication/325215555_DeepGlobe_2018_A_Challenge_to_Parse_the_Earth_through_Satellite_Images) uses DeepLab for baseline
    - [D-LinkNet](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf): dilated convolutions, resnet pretrained on imagenet, adam, BCE + dice loss, data augmentation by flips  
    - [FCN for Road Extraction](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Buslaev_Fully_Convolutional_Network_CVPR_2018_paper.pdf): unet with pretrained resnet as encoder, random contrast/brightness/HSV for augmentation, loss = 0.7BCE + 0.3log(Jaccard)
    - [Road Vectorization](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Filin_Road_Detection_With_CVPR_2018_paper.pdf): interesting algorithm for post processing

- [1.1](https://medium.com/the-downlinq/broad-area-satellite-imagery-semantic-segmentation-basiss-4a7ea2c8466f) and [1.2](https://medium.com/the-downlinq/creating-training-datasets-for-the-spacenet-road-detection-and-routing-challenge-6f970d413e2f)
    - Unet
    - they explain how to extract road masks from other data format
    - [code](https://github.com/CosmiQ/basiss)

- [2](https://blog.insightdatascience.com/deep-learning-for-disaster-recovery-45c8cd174d7a)
    - Adam 0.0001 LR
    - Dice Loss
    - Unet
    - Data augmentation through gaussian blur, flip and rotation at right angles

- [3](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
    - Unet
    - binary cross entropy
    - post-processing: parameterized operations on binarized outputs. morphology dilation/erosion to remove objects/holes smaller than a given threshold

- [4](https://www.cs.toronto.edu/~vmnih/docs/noisy_maps.pdf)
    - preprocess by standardization (helps normalize constrast)
    - cross entropy loss
    - I cannot find the datasets (URBAN1 and URBAN2)

- [5](https://www.cs.toronto.edu/~hinton/absps/road_detection.pdf)
    - very old
    - cross entropy loss
    - for postprocessing they train a NN on the predictions to match them to ground truth
    - they extract data from readily available satellite images
    
- [6](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Papers/Learning%20Aerial%20Image.pdf)
    - multinomial logistic loss
    - Fully Convolutional Network
    - large dataset automatically extracted from OpenStreetMap and Google Maps
    - SGD
    - data augmentation is very helpful

# ARCHITECTURES
- U-Net with SeResNext-50 or 101 as backbone
    - SeResNext101 is slightly better, maybe not worth it
    - upsampling in decoder instead of transposed convolutions
    - Adam with lr=0.0001
    - batch size = 4 shows good regularization, the alternative would be batch-size=8
    - 150 epochs, minima is usually reached around epoch 90 to 110
    - save best model on val_loss
    - performance depends quite a bit on the validation split

# PREPROCESSING
- standardization
- horizontal/vertical/diagonal flips, rotations, random contrast/brightness/HSV, Gaussian Blur

# ADDITIONAL DATA
- mine data from GMaps
- use Jonathan's images
- use SpaceNet dataset
- other sources (ignore for now):
    - mine data from [openstreetmap](https://help.openstreetmap.org/questions/44378/obtaining-unlabeled-road-data-layer) (should not be too hard)
    - mine data from [mapbox](https://docs.mapbox.com/vector-tiles/reference/mapbox-streets-v8/)
    - [Toronto City Dataset](http://www.cs.toronto.edu/~wenjie/papers/iccv17/wang_etal_iccv17.pdf)
    - [Massachussets Road Dataset](https://www.cs.toronto.edu/~vmnih/data/)
    - [SpaceNet Road (use Atlanta)](https://spacenetchallenge.github.io/datasets/spacenetRoads-summary.html)
    - [2D Semantic Labeling Contest (Postdam, Vaihingen)](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)

# POST PROCESSING
- [21/06 KMeans based post-processing colab code](https://colab.research.google.com/drive/1QWsI6bJnnN2MuU65U2AbGBXkj7fdjl0m?usp=sharing) Currently pulling from experiments branch
- CRF were not helpful
- Helpful to do morphological transformations (dilate+erode+dilate) to remove holes and noise. No thresholding.
- Han's 5D K-NN, maybe with more designed features, maybe in conjunction with a secondary network

# OTHER IDEAS
- 2016 literature survey covering every aspect of the semantic segmentation https://arxiv.org/pdf/1602.06541.pdf

# NOTEBOOKS
- [Best performing notebook from early June](https://colab.research.google.com/drive/11TNtlbcO_8kfSW39JXHiHJcWpIZ3NQWS?usp=sharing)
- [Best performing notebook from June 18](https://colab.research.google.com/drive/1n9rgCBDHuTttykR5Fz6JNRiEO253iIu2?usp=sharing) (after some tuning)
- [Best performing notebook from June 22](https://colab.research.google.com/drive/12BbjdJz_upR8Q2Ta5bCH24lO0844VcnB?usp=sharing) (after dropping the library model)

# COMMENTS
- test data has lots of parkings
- total entropy is not a good measure for the quality of the output (similar values for ground truth and bad output)
- can get around 84% accuracy by predicting always black
