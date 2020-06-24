import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os
import sys
from PIL import Image
import pickle

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_dilation

MASK_2_BINARY_THRESHOLD = 0.25
N_FEATS = 6
#N_CLUSTERS = 50
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
        self.N_CLUSTERS = 0
        self.BM_IN_AREA_PREC_THRESHOLD = BM_IN_AREA_PREC_THRESHOLD
        self.img_type = IMG_TYPE

        self.small_cluster_threshold = 0

        #variables for each image
        self.n_data_train = 0
        self.n_data_pred = 0
        self.n_feats = self.N_FEATS
        self.img_shape = (0,0)
        self.bm = np.array([])
        self.bm_near = np.array([])
        self.scaler_xy = StandardScaler()
        self.scaler_hsv = StandardScaler()

        self.test = "hello1"
        
    def gen_binary_map(self,mask):
        if np.max(mask) == 255:
            prob_map = mask/255.0
        else:
            raise(Exception)
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
        feats_train_hsv_scaled = self.scaler_hsv.transform(feats_train_hsv)
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
        y_train = km.fit_predict(feats_train_scaled)
        return(km)

    def gen_kmeans_scores_sklearn(self,km,feats_pred_scaled,use_bm_near=True):
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

    def gen_kmeans_scores_custom(self,km,feats_pred_scaled,use_bm_near=True):
        self.n_data_pred = feats_pred_scaled.shape[0]
        scores = np.zeros(self.n_data_pred)
        y_pred = km.predict(feats_pred_scaled)
        if use_bm_near:
            i=0
            for y in range(self.img_shape[0]):
                for x in range(self.img_shape[1]):
                    if self.bm_near[y][x] == 0:
                        scores[i] = np.nan
                    else:
                        v_c = feats_pred_scaled[i] - km.cluster_centers_[y_pred[i]]
                        scores[i] = np.dot(v_c,v_c)
                    i+=1
        else:
            i=0
            for y in range(self.img_shape[0]):
                for x in range(self.img_shape[1]):
                    v_c = feats_pred_scaled[i] - km.cluster_centers_[y_pred[i]]
                    scores[i] = np.dot(v_c,v_c)
                    i+=1

        scores = -1.0*scores
        #scores = -1.0*np.sqrt(scores)
        return(scores)

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


    def optimize_score_threshold(self,scores,bm,target):
        start=np.abs(np.max(scores))
        end=np.abs(np.min(scores))
        n_points = int( 10 *np.log10(end/start))+1
        
        threshold_tries = -1.0*np.logspace(np.log10(start),np.log10(end),n_points,base=10)
        prec_res=np.zeros(len(threshold_tries))
        iou_res=np.zeros(len(threshold_tries))
        for i in range(len(threshold_tries)):    
            bmgs = (scores > threshold_tries[i]).astype('uint8')
            bmgs = bmgs.reshape((bm.shape[0],bm.shape[1]))
            #bmgs = KMPP_single_image.bm_fill_lakes(bmgs,self.small_cluster_threshold)
        
            intersection = np.sum(bm.flatten()*bmgs.flatten())
            union_tmp = bm.flatten()+bmgs.flatten()
            union_tmp[union_tmp>1] = 1
            union = np.sum(union_tmp)
        
            prec_res[i] = intersection/np.sum(bm.flatten())
            iou_res[i] = intersection/union

        if target==0:
            opt_target = iou_res
        elif target==1:
            opt_target = prec_res+iou_res
        elif target==2:
            opt_target = prec_res+(iou_res-np.min(iou_res))/(np.max(iou_res)-np.min(iou_res))
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



class KMeansPP:
    def __init__(self,img_dir,mask_dir,output_dir):
        self.MASK_2_BINARY_THRESHOLD = MASK_2_BINARY_THRESHOLD
        self.N_FEATS = N_FEATS
        #self.N_CLUSTERS = N_CLUSTERS
        self.BM_IN_AREA_PREC_THRESHOLD = BM_IN_AREA_PREC_THRESHOLD
        self.img_type = IMG_TYPE

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir

        self.img_paths = glob.glob(os.path.join(self.img_dir,'*.'+self.img_type))
        self.mask_paths = glob.glob(os.path.join(self.mask_dir,'*.'+self.img_type))
        
        sample_img = np.array(Image.open(self.img_paths[0]))
        self.img_shape = np.shape(sample_img)
        self.n_data_pred = self.img_shape[0]*self.img_shape[1]


    def run_single_image(self,img,mask):
        ksi = KMPP_single_image()
        
        bm = ksi.gen_binary_map(mask)
        ksi.bm = bm
        if ksi.img_shape == (0,0):
            ksi.img_shape = np.shape(bm)

        ksi.N_CLUSTERS = int(np.sum(bm)/25/25)

        ksi.n_data_train = np.sum(bm.flatten())
        ksi.n_data_pred = bm.shape[0]*bm.shape[1]

        feats_scaled = ksi.img_gen_feats(img,bm)
        algo = ksi.train_kmeans(feats_scaled['train'])

        road_width_est = ksi.get_road_width(algo)
        area_scale = np.sqrt(ksi.scaler_xy.var_[0])*np.sqrt(ksi.scaler_xy.var_[1])

        small_cluster_threshold=int((road_width_est**2)*area_scale)
        ksi.small_cluster_threshold = small_cluster_threshold
        bm_dilate_factor = int(road_width_est*np.sqrt(area_scale)) 
        ksi.bm_near = binary_dilation(bm,iterations=bm_dilate_factor) 

        #scores = ksi.gen_kmeans_scores(algo,feats_scaled['pred'])
        #scores = ksi.gen_kmeans_scores_sklearn(algo,feats_scaled['pred'])
        scores = ksi.gen_kmeans_scores_custom(algo,feats_scaled['pred'],use_bm_near=True)

        #scores = np.nan_to_num(scores,nan=np.nanmin(scores))
        if np.isnan(scores).any():
            arg_nans = np.argwhere(np.isnan(scores)).flatten()
            scores[arg_nans] = np.nanmin(scores)
        
        opt_score_threshold = ksi.optimize_score_threshold(scores,bm,target=0)
        bmgs = (scores > opt_score_threshold).astype('uint8')
        bmgs = bmgs.reshape((img.shape[0],img.shape[1]))
        
        bmgs = KMPP_single_image.bm_fill_lakes(bmgs,small_cluster_threshold)
        bmgs = KMPP_single_image.bm_flood_islands(bmgs,small_cluster_threshold)
        
        bmog = bm.copy()
        low_prec_threshold = self.BM_IN_AREA_PREC_THRESHOLD
        bmgs = KMPP_single_image.bm_flood_low_prec(bmgs,bmog,low_prec_threshold)
        return(bmgs)




    def run_whole_dir(self):
        file_idx = 0
        for img_path,mask_path in list(zip(self.img_paths,self.mask_paths)):
            file_name = os.path.split(img_path)[-1]
            
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            output = self.run_single_image(img,mask)

            output = output*255.0
            output = output.astype(np.uint8)
            plt.imsave(os.path.join(self.output_dir,file_name),output,cmap=cm.gray)

            message = str(file_idx)+": "+file_name+" run_whole_dir\n"
            sys.stdout.write(message)
            sys.stdout.flush()
            file_idx += 1




class KMDT_single_image:
    def __init__(self):
        self.MASK_2_BINARY_THRESHOLD = MASK_2_BINARY_THRESHOLD
        self.N_FEATS = N_FEATS
        self.N_CLUSTERS = 0
        self.BM_IN_AREA_PREC_THRESHOLD = BM_IN_AREA_PREC_THRESHOLD
        self.img_type = IMG_TYPE

        self.small_cluster_threshold = 0

        #variables for each image
        self.n_data_train = 0
        self.n_data_pred = 0
        self.n_feats = self.N_FEATS
        self.img_shape = (0,0)
        self.bm = np.array([])

        self.clusters_info = np.array([])

        self.test = "hello1"
        
    def gen_binary_map(self,mask):
        prob_map = mask/255.0
        bm = (prob_map > self.MASK_2_BINARY_THRESHOLD).astype('uint8')
        return(bm)


    def img_gen_feats_xy(self,img,use_bm=False,bm=None):
        if use_bm:
            n_data = np.sum(bm.flatten())
        else:
            n_data = img.shape[0]*img.shape[1]
        feats_xy = np.zeros((n_data,2))
        row_idx = 0
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                write = True
                if use_bm:
                    if bm[y][x] == 0:
                        write =  False
                if write:
                    feats_xy[row_idx][0] = x
                    feats_xy[row_idx][1] = y
                    row_idx+=1
        feats_xy = feats_xy.astype(np.float)
        return(feats_xy)


    def img_gen_feats_hsv(self,img,use_bm=False,bm=None):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_img)
        if use_bm:
            n_data = np.sum(bm.flatten())
        else:
            n_data = img.shape[0]*img.shape[1]
        feats_hsv = np.zeros((n_data,3))
        if use_bm:
            row_idx = 0
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if bm[y][x] == 1:
                        feats_hsv[row_idx][0] = h[y][x]
                        feats_hsv[row_idx][1] = s[y][x]
                        feats_hsv[row_idx][2] = v[y][x]
                        row_idx+=1
        else:
            feats_hsv[:,0] = h.flatten()
            feats_hsv[:,1] = s.flatten()
            feats_hsv[:,2] = v.flatten()
        feats_hsv = feats_hsv.astype(np.float)
        return(feats_hsv)


    def train_kmeans(self,feats_train_scaled):
        km = KMeans(
            n_clusters=self.N_CLUSTERS, init='random',
            n_init=10, max_iter=300, 
            tol=1e-04, random_state=0
        )
        y_train = km.fit_predict(feats_train_scaled)
        return(km)


    @staticmethod
    def gen_cc_mat(algo):
        cluster_cents = algo.cluster_centers_[:,:2]
        n_clusters = len(cluster_cents)
        
        cc_mat_tmp = np.zeros((n_clusters,n_clusters,2)) #cluster to cluster
        cc_mat_tmp[:,:,0] = np.array([cluster_cents[:,0],]*n_clusters).transpose()
        cc_mat_tmp[:,:,1] = np.array([cluster_cents[:,1],]*n_clusters).transpose()
        
        cc_mat_tmp[:,:,0] = cc_mat_tmp[:,:,0]-cluster_cents[:,0]
        cc_mat_tmp[:,:,1] = cc_mat_tmp[:,:,1]-cluster_cents[:,1]
        
        cc_mat_sqed = cc_mat_tmp[:,:,0]**2
        cc_mat_sqed += cc_mat_tmp[:,:,1]**2
        cc_mat = np.sqrt(cc_mat_sqed)
        return(cc_mat)

    @staticmethod
    def gen_clusters_info(algo,feats_xy_bm,feats_hsv_bm):
        y_km_bm = algo.predict(feats_xy_bm)
        
        clusters_img = -1.0*np.ones((400,400))
        for i in range(len(feats_xy_bm)):
            clusters_img[int(feats_xy_bm[i][1])][int(feats_xy_bm[i][0])] = y_km_bm[i]
        
        n_clusters = algo.cluster_centers_.shape[0]
        clusters_hsv = np.zeros((n_clusters,3))
        clusters_hsv_std = np.zeros((n_clusters,3))
        clusters_area = np.zeros(n_clusters)
        clusters_peri = np.zeros(n_clusters)
        for c in range(n_clusters):
            clusters_hsv[c] = np.mean(feats_hsv_bm[y_km_bm==c],axis=0)
            clusters_hsv_std[c] = np.std(feats_hsv_bm[y_km_bm==c],axis=0)
            clusters_area[c] = np.sum(y_km_bm==c)
            
            cluster_tmp = ((clusters_img==c)*255.0).astype('uint8')
            cnts = cv2.findContours(cluster_tmp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            clusters_peri[c] = cv2.arcLength(cnts[1][0],True)
        result = {}
        result['hsv'] = clusters_hsv
        result['hsv_std'] = clusters_hsv_std
        result['area'] = clusters_area
        result['peri'] = clusters_peri
        return(result)

    @staticmethod
    def walk_find_next(cc_mat,idx_now,idx_walked,n_to_find):
        idx_sugg = np.argsort(cc_mat[idx_now])
        idx_next_list = []
        for i in range(1,len(idx_sugg)):
            if not(idx_sugg[i] in idx_walked):
                idx_next_list.append(idx_sugg[i])
            if len(idx_next_list)==n_to_find:
                break
        if i==len(idx_sugg):
            raise(Exception)
        return(idx_next_list)

    def gen_gbdt_feats(self,algo,feats_xy,feats_hsv):
        n_near = 3
        n_combs = int( (n_near-1)*n_near/2 )
        n_data = len(feats_xy)
        n_clusters = algo.cluster_centers_.shape[0]
        
        feats = -1.0*np.ones((n_data,49))
        if n_clusters==0:
            return(feats)

        row_idx = 0
        features_count = 0
        for y in range(self.img_shape[0]):
            for x in range(self.img_shape[1]):

                v_c = algo.cluster_centers_-feats_xy[row_idx]
                
                m_c_sqed = (v_c[:,0])**2
                m_c_sqed += (v_c[:,1])**2
        
                idxs_near = np.argsort(m_c_sqed)[:n_near]
        
                ######
                v_hsv_diff = feats_hsv[row_idx]-self.clusters_info['hsv'][idxs_near[0]]
                v_hsv_diff_abs = np.abs(v_hsv_diff)
                m_hsv_diff = np.linalg.norm(v_hsv_diff)
                v_hsv_diff_scaled = v_hsv_diff/self.clusters_info['hsv_std'][idxs_near[0]]
                v_hsv_diff_scaled_abs = np.abs(v_hsv_diff_scaled)
                m_hsv_diff_scaled = np.linalg.norm(v_hsv_diff_scaled) 
                feats_row = np.concatenate((v_hsv_diff,
                                            v_hsv_diff_abs,
                                            v_hsv_diff_scaled,
                                            v_hsv_diff_scaled_abs))
                feats_row = np.append(feats_row,m_hsv_diff)
                feats_row = np.append(feats_row,m_hsv_diff_scaled)
                
                v_c_near = v_c[idxs_near]
                
                prox_mat = np.outer(v_c_near[:,0],v_c_near[:,0])
                prox_mat += np.outer(v_c_near[:,1],v_c_near[:,1])
                prox_mat = np.abs(prox_mat)
                
                ###
                xy_scaled = np.array([feats_xy[row_idx][0]/self.img_shape[1],
                                      feats_xy[row_idx][1]/self.img_shape[0]])
                feats_row = np.append(feats_row,xy_scaled)
                m_xy_scaled = np.linalg.norm(xy_scaled-0.5)
                feats_row = np.append(feats_row,m_xy_scaled)
                
                ###
                m_c_near = -1.0*np.ones(3)
                m_c_near[:len(idxs_near)] = (np.sqrt(np.diag(prox_mat)))[:len(idxs_near)]
                feats_row = np.append(feats_row,m_c_near)
                
                prox_mat_tmp = prox_mat/m_c_near
                prox_mat = (prox_mat_tmp.transpose())/m_c_near
                
                ###
                m_cc_near = -1.0*np.ones(3)
                if len(idxs_near)>=2:
                    m_cc_near[0] = self.cc_mat[idxs_near[0]][idxs_near[1]]
                if len(idxs_near)>=3:
                    m_cc_near[1] = self.cc_mat[idxs_near[0]][idxs_near[2]]
                    m_cc_near[2] = self.cc_mat[idxs_near[1]][idxs_near[2]]
                feats_row = np.append(feats_row,m_cc_near)
                feats_row = np.append(feats_row,m_c_near[0]/m_cc_near[0])
                
                ###
                prox_mat_triu = np.triu(prox_mat,k=1)
                cos_near = -1.0*np.ones(3)
                if len(idxs_near)>=2:
                    cos_near[0] = prox_mat_triu[0][1]
                if len(idxs_near)>=3:
                    cos_near[1] = prox_mat_triu[0][2]
                    cos_near[2] = prox_mat_triu[1][2]
                feats_row = np.append(feats_row,cos_near)
                
                sin_near = np.sqrt(1.0-cos_near**2)
                ###
                cross_near = -1.0*np.ones(3)
                if len(idxs_near)>=2:
                    cross_near[0] = prox_mat_triu[0][1]*m_c_near[0]*m_c_near[1]
                if len(idxs_near)>=3:
                    cross_near[1] = prox_mat_triu[0][2]*m_c_near[0]*m_c_near[2]
                    cross_near[2] = prox_mat_triu[1][2]*m_c_near[1]*m_c_near[2]
                feats_row = np.append(feats_row,cross_near)
        
                ###
                area_near = -1.0*np.ones(3)
                area_near[:len(idxs_near)] = self.clusters_info['area'][idxs_near]
                feats_row = np.append(feats_row,area_near)
                peri_near = -1.0*np.ones(3)
                peri_near[:len(idxs_near)] = self.clusters_info['peri'][idxs_near]
                feats_row = np.append(feats_row,peri_near)
                adp_near = -1.0*np.ones(3)
                adp_near[:len(idxs_near)] = area_near[:len(idxs_near)]/((peri_near+1.0)[:len(idxs_near)])
                feats_row = np.append(feats_row,adp_near)
                
                ###
                mp_near = -1.0*np.ones(3)
                mp_near[:len(idxs_near)] = (cross_near/m_cc_near)[:len(idxs_near)]
                feats_row = np.append(feats_row,mp_near)
                mpdexp_near = -1.0*np.ones(3)
                mpdexp_near[:len(idxs_near)] = (mp_near/(area_near/np.min(m_cc_near)))[:len(idxs_near)]
                feats_row = np.append(feats_row,mpdexp_near)
                
                road_metric = -1.0*np.ones(2)
                ### metric 5
                #distance penalty
                if len(idxs_near)>=3:
                    decay = 100
                    prox_mat_tmp = prox_mat*np.exp(-m_c_near/(2*decay))
                    prox_mat_5 = (prox_mat_tmp.transpose())*np.exp(-m_c_near/(2*decay))        
                    road_metric[0] = np.sum(np.triu(prox_mat_5,k=1))/n_combs
        
                ###
                if n_clusters>5:
                    n_steps = 5
                    idx_init = idxs_near[0]
                    m_walked = 0
                    idx_walked = [idx_init]
                    idx_now = idx_init
                    idx_now2 = idx_init
                    for i in range(0,n_steps):
                        if i == 0:
                            idx_next_list = KMDT_single_image.walk_find_next(
                                    self.cc_mat,idx_now,idx_walked,2)
                            idx_next = idx_next_list[0]
                            idx_next2 = idx_next_list[1]
                        else:
                            idx_next = KMDT_single_image.walk_find_next(
                                    self.cc_mat,idx_now,idx_walked,1)[0]
                            if self.cc_mat[idx_now][idx_next]>self.cc_mat[idx_now2][idx_next2]:
                                idx_now_tmp=idx_now
                                idx_now=idx_now2
                                idx_now2=idx_now_tmp
        
                                idx_next_tmp=idx_next
                                idx_next=idx_next2
                                idx_next2=idx_next_tmp
        
                        m_walked += self.cc_mat[idx_now][idx_next]
                        idx_walked.append(idx_next)
                        idx_now = idx_next
                    road_metric[1] = m_walked
                else:
                    road_metric[1] = 0.0
                feats_row = np.append(feats_row,road_metric)
                
                feats_row = np.append(feats_row,np.sum(self.clusters_info['area']))
                feats_row = np.append(feats_row,n_clusters)
        
                features_count = len(feats_row)
                feats[row_idx] = feats_row
                #if row_idx%10000==0:
                #    message = str(row_idx)+": "+" gen_gbdt_feats\n"
                #    sys.stdout.write(message)
                #    sys.stdout.flush()
                row_idx+=1
        return(feats)


class KMDTPP:
    def __init__(self,img_dir,mask_dir,output_dir):
        self.MASK_2_BINARY_THRESHOLD = MASK_2_BINARY_THRESHOLD
        self.N_FEATS = N_FEATS
        #self.N_CLUSTERS = N_CLUSTERS
        self.BM_IN_AREA_PREC_THRESHOLD = BM_IN_AREA_PREC_THRESHOLD
        self.img_type = IMG_TYPE

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.km_model_dir = ''

        self.img_paths = glob.glob(os.path.join(self.img_dir,'*.'+self.img_type))
        self.mask_paths = glob.glob(os.path.join(self.mask_dir,'*.'+self.img_type))
        
        sample_img = np.array(Image.open(self.img_paths[0]))
        self.img_shape = np.shape(sample_img)
        self.n_data_pred = self.img_shape[0]*self.img_shape[1]



    def gen_km_models(self,km_model_dir):
        file_idx = 0
        self.km_model_dir = km_model_dir
        for img_path,mask_path in list(zip(self.img_paths,self.mask_paths)):
            file_name = os.path.split(img_path)[-1]
            model_name = file_name.split(".")[0] + '.pkl'
            
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            kdsi = KMDT_single_image()
            
            bm = kdsi.gen_binary_map(mask)
            kdsi.bm = bm
            if kdsi.img_shape == (0,0):
                kdsi.img_shape = np.shape(bm)
            
            n_data = kdsi.img_shape[0]*kdsi.img_shape[1]
            n_data_bm = np.sum(kdsi.bm) #equivalent to area
            kdsi.N_CLUSTERS = int(n_data_bm/25/25)
            
            feats_xy_km = kdsi.img_gen_feats_xy(img,use_bm=True,bm=bm)
            algo = kdsi.train_kmeans(feats_xy_km)

            model_save_path = os.path.join(km_model_dir,model_name)
            pickle.dump( algo,open(model_save_path,'wb') )

            message = str(file_idx)+": "+" gen_kmeans models\n"
            sys.stdout.write(message)
            sys.stdout.flush()
            file_idx += 1

    def gen_gbdt_feats(self,km_model_dir,feats_dir):
        file_idx = 0
        self.km_model_dir = km_model_dir
        if not(os.path.exists(feats_dir)):
            os.path.mkdir(feats_dir)

        for img_path,mask_path in list(zip(self.img_paths,self.mask_paths)):
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            file_name = os.path.split(img_path)[-1]
            km_model_name = file_name.split(".")[0] + ".pkl"
            km_model_path = os.path.join(km_model_dir,km_model_name)
            feats_name = file_name.split(".")[0] + ".npy"
            feats_path = os.path.join(feats_dir,feats_name)

            kdsi = KMDT_single_image()
            bm = kdsi.gen_binary_map(mask)
            kdsi.bm = bm
            if kdsi.img_shape == (0,0):
                kdsi.img_shape = np.shape(bm)
            y_road = kdsi.bm.flatten()
            
            algo = pickle.load(open(km_model_path,"rb"))
            
            feats_xy = kdsi.img_gen_feats_xy(img)
            feats_hsv = kdsi.img_gen_feats_hsv(img)
            feats_xy_bm = kdsi.img_gen_feats_xy(img,use_bm=True,bm=bm)
            feats_hsv_bm = kdsi.img_gen_feats_hsv(img,use_bm=True,bm=bm)
            
            kdsi.clusters_info = KMDT_single_image.gen_clusters_info(algo,
                    feats_xy_bm,feats_hsv_bm)
            kdsi.cc_mat = KMDT_single_image.gen_cc_mat(algo)
            feats_xgb = kdsi.gen_gbdt_feats(algo,feats_xy,feats_hsv)

            with open(feats_path,'wb') as f:
                np.save(f,feats_xgb)

            message = str(file_idx)+": "+" gen_gbdt_feats\n"
            sys.stdout.write(message)
            sys.stdout.flush()
            file_idx += 1

    def run_whole_dir(self):
        file_idx = 0
        for img_path,mask_path in list(zip(self.img_paths,self.mask_paths)):
            file_name = os.path.split(img_path)[-1]
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))


###
            output = self.run_single_image(img,mask)

            output = output*255.0
            output = output.astype(np.uint8)
            plt.imsave(os.path.join(self.output_dir,file_name),output,cmap=cm.gray)

            message = str(file_idx)+": "+file_name+" run_whole_dir\n"
            sys.stdout.write(message)
            sys.stdout.flush()
            file_idx += 1

