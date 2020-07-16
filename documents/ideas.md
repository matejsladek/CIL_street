# TODO:

- for post processing, find best number of iterations (2 parameters)
- find right parameter for ensemble
- last cv runs for the report
	- table 2
	- table 3 (need cross validated f1, accuracy, patch wise accuracy and iou on the two baselines and on ensemble)
- finish up report
- add requirements.txt

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

# Reporting and Experiments
- Papers for reference: 
    - Road Seg Specific: Henry 2018, Hinton&Mnih
- Training
    - Tensorflow version, GPU type, single (vs parallel)
    - Hyperparameter table
    - Rough convergence time
- Metrics
    - Main: IoU/Jaccard index, precision, recall (Commonly used metrics in literature are meaningful: facilitate comparison)
    - Others: F1, Dice coef, Hausdorff distance, sensitivity, specificity
    - Cross validated above: k=5,
    - Hinton&Mnih: PR curve with buffer/patches- standard protocol
    - Comment: accuracy is not illustrative

- Metrics to report (according to Marco):
	- from 5 fold CV: Accuracy, F1 score, patch-wise accuracy
	- public test score on kaggle

- Experiments
    - Preprocess vs no preprocess (baseline) (Report IoU)
    - original 90 vs original 90 + chicago 1800 (baseline) (Report IoU)
    - Proceed with one combination: e.g. preprocess + original 90 + chicago 1800
    - Briefly: Postprocess vs no postprocess
    
    - Discuss effect of changing: domain specific HPs, algo HPs

    - Hyperparameter tuning using CV
    - Threshold selection using PR curve
    - Use CVed loss vs epochs plot to justify epochs trained, plot minimum as rigourous convergence measure

- Models for comparison:
	- Baseline #1 from programming exercises
	- Baseline #2 from programming exercises
	- resnet101 based unet with original data, no postor preprocessing
	- resnet101 based unet with additional data and preprocessing
	- add pretraining
	- add squeeze and excitation to encoder (and maybe decoder) (seresnext101 backbone)
	- enable MTL
	- use an ensemble
	- enable post process

all steps should increase at least the patch wise accuracy expect maybe MTL, but hopefully that improves something else

Open questions:
	- what kind of MTL? how do we weight the losses? I think contour + reduced weight for secondary task is the way to go, but we would need to check
	- what parameters for postprocessing? we do morphological transformations, but iteration number needs to be set
	- how many networks in the ensemble?
	- what parameters for random brightness/contrast?
    
A:
        - Loss weight tuning: 1,2/1,2,4,8 for each (see Henry 2018 table)
	- Morphological PP param: tune using PR curve?

# COMMENTS
- test data has lots of parkings
- total entropy is not a good measure for the quality of the output (similar values for ground truth and bad output)
- can get around 84% accuracy by predicting always black
