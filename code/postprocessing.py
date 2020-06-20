import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_dilation

MASK_2_BINARY_THRESHOLD = 0.25
N_FEATS = 6
N_CLUSTERS = 50
BM_IN_AREA_PREC_THRESHOLD = 0.25
IMG_TYPE = 'png'

def simple_dilate_erode():
    img_paths = glob.glob('output/*.png')
    for img_path in img_paths:
        img = cv2.imread(img_path, 0)
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=3)
        img = cv2.erode(img, kernel, iterations=8)
        img = cv2.dilate(img, kernel, iterations=3)
        cv2.imwrite(img_path.replace('output', '3x3_dilate3_erode8_dilate3'), img)

        

class KMPP_single_image:
    def __init__(self):
        self.MASK_2_BINARY_THRESHOLD = MASK_2_BINARY_THRESHOLD
        self.N_FEATS = N_FEATS
        self.N_CLUSTERS = N_CLUSTERS
        self.BM_IN_AREA_PREC_THRESHOLD = BM_IN_AREA_PREC_THRESHOLD
        self.img_type = IMG_TYPE

        #variables for each image
        self.n_data_train = 0
        self.n_data_pred = 0
        self.n_feats = self.N_FEATS
        self.bm = np.array([])
        self.bm_near = np.array([])
        self.scaler_xy = StandardScaler()
        self.scaler_hsv = StandardScaler()
        #self.feats_train_scaled = np.array([])
        #self.feats_pred_scaled = np.array([])
        
        #self.km = KMeans()
        #self.bmgs


    def gen_binary_map(self,mask):
        prob_map = mask/255.0
        bm = (prob_map > self.MASK_2_BINARY_THRESHOLD).astype('uint8')
        return(bm)

    def img_gen_feats(self,img,bm):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_img)
        
        ###
        feats_train_xy = np.zeros((self.n_data_train,2))
        feats_train_hsv = np.zeros((self.n_data_train,3))
        feats_pred_xy = np.zeros((self.n_data_pred,2))
        feats_pred_hsv = np.zeros((self.n_data_pred,3))
        
        row_idx_train = 0
        row_idx_pred = 0
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                feats_pred_xy[row_idx_pred][0] = x
                feats_pred_xy[row_idx_pred][1] = y
                row_idx_pred+=1
                if bm[y][x] == 1:
                    feats_train_xy[row_idx_train][0] = x
                    feats_train_xy[row_idx_train][1] = y
                    
                    feats_train_hsv[row_idx_train][0] = h[y][x]
                    feats_train_hsv[row_idx_train][1] = s[y][x]
                    feats_train_hsv[row_idx_train][2] = v[y][x]
                    
                    row_idx_train+=1
                    
        feats_pred_hsv[:,0] = h.flatten()
        feats_pred_hsv[:,1] = s.flatten()
        feats_pred_hsv[:,2] = v.flatten()
        
        feats_train_xy = feats_train_xy.astype(np.float)
        feats_train_hsv = feats_train_hsv.astype(np.float)
        feats_pred_xy = feats_pred_xy.astype(np.float)
        feats_pred_hsv = feats_pred_hsv.astype(np.float)		
        
        ###
        feats_train_scaled = np.zeros((self.n_data_train,self.n_feats))
        feats_pred_scaled = np.zeros((self.n_data_pred,self.n_feats))
        
        self.scaler_xy = StandardScaler()
        self.scaler_xy.fit(feats_train_xy)
        self.scaler_hsv = StandardScaler()
        self.scaler_hsv.fit(feats_train_hsv)
        
        feats_train_xy_scaled = self.scaler_xy.transform(feats_train_xy)
        #feats_train_hsv: trickly set to zeros. Untouched
        #scale the pred hsvs by the train scaler for consistency
        feats_pred_xy_scaled = self.scaler_xy.transform(feats_pred_xy)
        feats_pred_hsv_scaled = self.scaler_hsv.transform(feats_pred_hsv)
        
        feats_train_scaled[:,0] = feats_train_xy_scaled[:,0]
        feats_train_scaled[:,1] = feats_train_xy_scaled[:,1]
        feats_pred_scaled[:,0] = feats_pred_xy_scaled[:,0]
        feats_pred_scaled[:,1] = feats_pred_xy_scaled[:,1]
        feats_pred_scaled[:,2] = feats_pred_hsv_scaled[:,0]
        feats_pred_scaled[:,3] = feats_pred_hsv_scaled[:,1]
        feats_pred_scaled[:,4] = feats_pred_hsv_scaled[:,2]
        return({'train':feats_train_scaled,'pred':feats_pred_scaled})

    def train_kmeans(self,feats_train_scaled):
        km = KMeans(
            n_clusters=self.N_CLUSTERS, init='random',
            n_init=10, max_iter=300, 
            tol=1e-04, random_state=0
        )
        y_km = km.fit_predict(feats_train_scaled)
        return(km)

    def gen_kmeans_scores(self,km,feats_pred_scaled,use_bm_near=True):
        self.n_data_pred = feats_pred_scaled.shape[0]
        scores = np.zeros(self.n_data_pred)
        if use_bm_near:
            i = 0
            for y in range(self.bm_near.shape[0]):
                for x in range(self.bm_near.shape[1]):
                    if self.bm_near[y][x] == 0:
                        scores[i] = np.nan
                    else:
                        scores[i] = km.score(feats_pred_scaled[i:i+1])
                    i+=1
            #scores = np.nan_to_num(scores,nan=np.nanmin(scores))
        else:
            for i in range(self.n_data_pred):
                scores[i] = km.score(feats_pred_scaled[i:i+1])
        return(scores)

    def optimize_score_threshold(self,scores,bm):
        start=.01
        end=100
        n_points = int( 4 *np.log10(end/start))+1
        start=np.log10(start)
        end=np.log10(end)
        
        threshold_tries = -1.0*np.logspace(start,end,n_points,base=10)
        prec_res=np.zeros(len(threshold_tries))
        iou_res=np.zeros(len(threshold_tries))
        for i in range(len(threshold_tries)):    
            bmgs = (scores > threshold_tries[i]).astype('uint8')
            bmgs = bmgs.reshape((bm.shape[0],bm.shape[1]))
        
            intersection = np.sum(bm.flatten()*bmgs.flatten())
            union_tmp = bm.flatten()+bmgs.flatten()
            union_tmp[union_tmp>1] = 1
            union = np.sum(union_tmp)
        
            prec_res[i] = intersection/np.sum(bm.flatten())
            iou_res[i] = intersection/union
            
        opt_target=prec_res+(iou_res-np.min(iou_res))/(np.max(iou_res)-np.min(iou_res))
        threshold_opt = threshold_tries[np.argmax(opt_target)]
        return(threshold_opt)

    @staticmethod
    def get_road_len(cc_mat,n_clusters):
        cc_mat_flat = cc_mat.flatten()
        idxs = np.argsort(cc_mat_flat)[n_clusters:n_clusters*2]
        len_est = np.sum(cc_mat_flat[idxs])
        return(len_est)

    def get_road_width(self,km):
        cluster_cents = km.cluster_centers_[:,:2]
        n_clusters = len(cluster_cents)
        
        cc_mat_tmp = np.zeros((n_clusters,n_clusters,2)) #cluster to cluster
        cc_mat_tmp[:,:,0] = np.array([cluster_cents[:,0],]*n_clusters).transpose()
        cc_mat_tmp[:,:,1] = np.array([cluster_cents[:,1],]*n_clusters).transpose()
        
        cc_mat_tmp[:,:,0] = cc_mat_tmp[:,:,0]-cluster_cents[:,0]
        cc_mat_tmp[:,:,1] = cc_mat_tmp[:,:,1]-cluster_cents[:,1]
        
        cc_mat_sqed = cc_mat_tmp[:,:,0]**2
        cc_mat_sqed += cc_mat_tmp[:,:,1]**2
        
        cc_mat = np.sqrt(cc_mat_sqed)
        
        area_scale = np.sqrt(self.scaler_xy.var_[0])*np.sqrt(self.scaler_xy.var_[1])
        road_length_est = KMPP_single_image.get_road_len(cc_mat,n_clusters)
        road_width_est = self.n_data_train/area_scale/road_length_est
        return(road_width_est)

    @staticmethod
    def bm_get_clusters_info(bm):
        lw, num = measurements.label(bm)
        area = measurements.sum(bm, lw, index=np.arange(lw.max() + 1))
        return((lw,area))

    @staticmethod
    def bm_fill_lakes(bm,small_cluster_threshold):
        bm_neg = 1-bm
        lw,area = KMPP_single_image.bm_get_clusters_info(bm_neg)
        label_land = np.argwhere(area>small_cluster_threshold)
        bm_filled = np.ones((bm.shape[0],bm.shape[1]))
        for y in range(bm.shape[0]):
            for x in range(bm.shape[1]):
                if lw[y][x] in label_land:
                    bm_filled[y][x] = 0
        return(bm_filled)
    
    @staticmethod
    def bm_flood_islands(bm,small_cluster_threshold):
        lw,area = KMPP_single_image.bm_get_clusters_info(bm)
        labels_isls = np.argwhere(area>small_cluster_threshold)
        bm_flooded = np.zeros((bm.shape[0],bm.shape[1]))
        for y in range(bm.shape[0]):
            for x in range(bm.shape[1]):
                if lw[y][x] in labels_isls:
                    bm_flooded[y][x] = 1
        return(bm_flooded)
    
    @staticmethod
    def bm_flood_low_prec(bm,bmog,low_prec_threshold):
        lw,area = KMPP_single_image.bm_get_clusters_info(bm)
        labels_land = np.arange(len(area))
        bmog_in_area = np.zeros(len(area))
        
        for y in range(bm.shape[0]):
            for x in range(bm.shape[1]):
                if lw[y][x] != 0:
                    if bmog[y][x] == 1:
                        bmog_in_area[lw[y][x]]+=1
        
        area_tmp=area.copy()
        area_tmp[0]=1
        bmog_in_area_prec = bmog_in_area/area_tmp
        
        labels_bm_high_prec = np.argwhere(bmog_in_area_prec>low_prec_threshold).flatten()
        bm_high_prec = np.zeros((bm.shape[0],bm.shape[1]))
        for y in range(bm.shape[0]):
            for x in range(bm.shape[1]):
                if lw[y][x] in labels_bm_high_prec:
                    bm_high_prec[y][x] = 1
        return(bm_high_prec)


class KMeansPP:
    def __init__(self,img_dir,mask_dir,output_dir):
        self.MASK_2_BINARY_THRESHOLD = MASK_2_BINARY_THRESHOLD
        self.N_FEATS = N_FEATS
        self.N_CLUSTERS = N_CLUSTERS
        self.BM_IN_AREA_PREC_THRESHOLD = BM_IN_AREA_PREC_THRESHOLD
        self.img_type = IMG_TYPE

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir

        #self.img_paths = []
        #self.mask_paths = []
        
        #self.img_shape = (0,0)
        #self.n_data_pred = 0

        self.img_paths = glob.glob(os.path.join(self.img_dir,'*.'+self.img_type))
        self.mask_paths = glob.glob(os.path.join(self.mask_dir,'*.'+self.img_type))
        
        sample_img = np.array(Image.open(self.img_paths[0]))
        self.img_shape = np.shape(sample_img)
        self.n_data_pred = self.img_shape[0]*self.img_shape[1]

    def run_single_image(self,img,mask):
        ksi = KMPP_single_image()
        
        bm = ksi.gen_binary_map(mask)

        ksi.n_data_train = np.sum(bm.flatten())
        ksi.n_data_pred = bm.shape[0]*bm.shape[1]

        feats_scaled = ksi.img_gen_feats(img,bm)
        algo = ksi.train_kmeans(feats_scaled['train'])
        scores = ksi.gen_kmeans_scores(algo,feats_scaled['pred'])
        
        opt_score_threshold = ksi.optimize_score_threshold(scores,bm)
        bmgs = (scores > opt_score_threshold).astype('uint8')
        bmgs = bmgs.reshape((img[0],img[1]))

        road_width_est = ksi.get_road_width(algo)
        area_scale = np.sqrt(ksi.scaler_xy.var_[0])*np.sqrt(ksi.scaler_xy.var_[1])
        small_cluster_threshold=int((road_width_est**2)*area_scale)
        
        bmgs = KMPP_single_image.bm_fill_lakes(bmgs,small_cluster_threshold)
        bmgs = KMPP_single_image.bm_flood_islands(bmgs,small_cluster_threshold)
        
        bmog = bm.copy()
        low_prec_threshold = self.BM_IN_AREA_PREC_THRESHOLD
        bmgs = KMPP_single_image.bm_flood_low_prec(bmgs,bmog,low_prec_threshold)
        return(bmgs)

    def run_single_image2(self,img,mask):
        ksi = KMPP_single_image()
        
        bm = ksi.gen_binary_map(mask)
        ksi.bm = bm

        ksi.n_data_train = np.sum(bm.flatten())
        ksi.n_data_pred = bm.shape[0]*bm.shape[1]

        feats_scaled = ksi.img_gen_feats(img,bm)
        algo = ksi.train_kmeans(feats_scaled['train'])

        road_width_est = ksi.get_road_width(algo)
        area_scale = np.sqrt(ksi.scaler_xy.var_[0])*np.sqrt(ksi.scaler_xy.var_[1])

        small_cluster_threshold=int((road_width_est**2)*area_scale)
        bm_dilate_factor = int(road_width_est*np.sqrt(area_scale)) 
        ksi.bm_near = binary_dilation(bm,iterations=bm_dilate_factor) 

        scores = ksi.gen_kmeans_scores(algo,feats_scaled['pred'],use_bm_near=True)
        scores = np.nan_to_num(scores,nan=np.nanmin(scores))
        
        opt_score_threshold = ksi.optimize_score_threshold(scores,bm)
        bmgs = (scores > opt_score_threshold).astype('uint8')
        bmgs = bmgs.reshape((img[0],img[1]))
        
        bmgs = KMPP_single_image.bm_fill_lakes(bmgs,small_cluster_threshold)
        bmgs = KMPP_single_image.bm_flood_islands(bmgs,small_cluster_threshold)
        
        bmog = bm.copy()
        low_prec_threshold = self.BM_IN_AREA_PREC_THRESHOLD
        bmgs = KMPP_single_image.bm_flood_low_prec(bmgs,bmog,low_prec_threshold)
        return(bmgs)

    def run_single_image3(self,img,mask):
        ksi = KMPP_single_image()
        
        bm = ksi.gen_binary_map(mask)
        ksi.bm = bm

        ksi.n_data_train = np.sum(bm.flatten())
        ksi.n_data_pred = bm.shape[0]*bm.shape[1]

        feats_scaled = ksi.img_gen_feats(img,bm)
        algo = ksi.train_kmeans(feats_scaled['train'])

        road_width_est = ksi.get_road_width(algo)
        area_scale = np.sqrt(ksi.scaler_xy.var_[0])*np.sqrt(ksi.scaler_xy.var_[1])

        small_cluster_threshold=int((road_width_est**2)*area_scale)
        bm_dilate_factor = int(road_width_est*np.sqrt(area_scale)) 
        ksi.bm_near = binary_dilation(bm,iterations=bm_dilate_factor) 

        scores = ksi.gen_kmeans_scores(algo,feats_scaled['pred'],use_bm_near=True)
        scores = np.nan_to_num(scores,nan=np.nanmin(scores))
        
        opt_score_threshold = ksi.optimize_score_threshold(scores,bm)
        bmgs = (scores > opt_score_threshold).astype('uint8')
        bmgs = bmgs.reshape((img.shape[0],img.shape[1]))
        
        bmgs = KMPP_single_image.bm_fill_lakes(bmgs,small_cluster_threshold)
        bmgs = KMPP_single_image.bm_flood_islands(bmgs,small_cluster_threshold)
        
        bmog = bm.copy()
        low_prec_threshold = self.BM_IN_AREA_PREC_THRESHOLD
        bmgs = KMPP_single_image.bm_flood_low_prec(bmgs,bmog,low_prec_threshold)
        return(bmgs)


    def run_whole_dir(self):
        for img_path,mask_path in list(zip(self.img_paths,self.mask_paths)):
            name = os.path.split(img_path)[-1]
            
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            #output = self.run_single_image(img,mask)
            #output = self.run_single_image2(img,mask)
            output = self.run_single_image3(img,mask)

            output = output.astype(np.uint8)
            output_img = tf.keras.preprocessing.image.array_to_img(output)
            output_img.save(os.path.join(self.output_dir,name))
    
    def run_whole_dir2(self):
        file_idx = 0
        for img_path,mask_path in list(zip(self.img_paths,self.mask_paths)):
            file_name = os.path.split(img_path)[-1]
            
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            output = self.run_single_image3(img,mask)

            ###
            #output_img = tf.keras.preprocessing.image.array_to_img(output).resize((608,608))
            #output_img.save(os.path.join(self.output_dir,file_name))

            ###
            import matplotlib.cm as cm
            output = output*255.0
            output = output.astype(np.uint8)
            plt.imsave(os.path.join(self.output_dir,file_name),output,cmap=cm.gray)

            message = str(file_idx)+": "+file_name+" run_whole_dir\n"
            sys.stdout.write(message)
            sys.stdout.flush()
            file_idx += 1


