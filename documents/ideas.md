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
- currently using U-Net with pretrained seResNext50
    - choose backbone (Marco)
    - need to rimplement it (Marco)
    - need to tune it (hyperparameters and architecture (dilation and aspp?)) (Marco)
- MTL architecture using Sobel edge detection OR derivatives, if successful expand to multi label segmentation


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
- try [conditional random fields](https://github.com/lucasb-eyer/pydensecrf/blob/master/examples/Non%20RGB%20Example.ipynb)
 (idea: scan image to find angle with the most streets, use that as a prior, or simply use vertical/horizontal)
- something based on canny edge detection although is probably useless
[link1](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123), 
[link2](http://www.sci.utah.edu/~cscheid/spr05/imageprocessing/project4/)
- hardcoded filter e.g. if column is mostly one, set it to 1
- pass with a convolutional filter filling in holes and i.e. if line to the left is full and we have a hole, fill the hole
- something based on Hough transform
- parameterized operations on binarized outputs. morphology dilation/erosion to remove objects/holes smaller than a given threshold
- secondary NN


# OTHER IDEAS
- double network with ce/surface loss + final network for refining predictions (separate or end to end training)


# NOTEBOOKS
- [Initial colab by Han](https://colab.research.google.com/drive/14Cs7Bs1DXQCTGUOj-cViJiKC47O_E2cA)
- [U-net with tf.data](https://drive.google.com/open?id=1EgznF_kmUdJmsT0qDfY2tKxLHOZrsMdi)
- [Simplified U-Net](https://drive.google.com/open?id=11Tx38SgUgQSCccHkl6tC7K13YFN7fd1b)
- [Try using pretrained backbones](https://colab.research.google.com/drive/1IruU8ALJoiqHP1FqDNqtt2EpxD8Vwgpe?usp=sharing)
- [Best performing notebook](https://colab.research.google.com/drive/11TNtlbcO_8kfSW39JXHiHJcWpIZ3NQWS?usp=sharing)
- [Multi task learning experiment](https://colab.research.google.com/drive/1cS6_Y5TXMQnVFMGdsNavb1Nh-N_fMowI?usp=sharing)


#TODO
- Matej: experiment with additional data, see which dataset works best
- Han: post processing
- Jonathan: check out multi task learning
- Marco: reimplement and tune the model


# COMMENTS
- test data has lots of parkings
- total entropy is not a good measure for the quality of the output (similar values for ground truth and bad output)
- can get around 80% accuracy by predicting always black
- it is very hard to adapt pretrained networks. Bigger unets (resnet50 backbone) are also significantly less stable.