# -----------------------------------------------------------
# Helper script
# Generate experiment configs organized in a folder from a
# specified baseline.
# Requires manual definition of config diffs in functions.
# Examples below.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import json
import os
import argparse
import shutil

project_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),"..",".."])

def diff_to_conf(base,exp_diff_list,exp_group_diff):
    config_list = []
    for exp_diff in exp_diff_list:
        config_tmp = base.copy()
        for k,v in exp_group_diff.items():
            config_tmp[k] = v
        for k,v in exp_diff.items():
            config_tmp[k] = v
        config_list.append(config_tmp)
    return(config_list)


def gen_decoders1907A_configs(base):
#    exp_group_name = 'decoders1907A_debug'
    exp_group_name = 'decoders1907A_maps1800'
    exp_group_diff = {
            'name':os.path.join(exp_group_name,'exp'),
            'experimental_decoder':True,
            'decoder_exp_setting':'A',
            'cv_k_todo':1
            }
    exp_diff_list = [
            {'residual':False,'art':False,'se':False},
            {'residual':True,'art':False,'se':False},
            {'residual':False,'art':True,'se':False},
            {'residual':False,'art':False,'se':True},
            {'residual':False,'art':True,'se':True},
            {'residual':True,'art':True,'se':True},
            {'experimental_decoder':False}
            ]
    config_list = diff_to_conf(base,exp_diff_list,exp_group_diff)
    return config_list

def gen_decoders1907B_configs(base):
#    exp_group_name = 'decoders1907B_debug'
    exp_group_name = 'decoders1907B_maps1800'
    exp_group_diff = {
            'name':os.path.join(exp_group_name,'exp'),
            'experimental_decoder':True,
            'decoder_exp_setting':'B',
            'cv_k_todo':1
            }
    exp_diff_list = [
            {'residual':False,'art':False,'se':False},
            {'residual':True,'art':False,'se':False},
            {'residual':False,'art':True,'se':False},
            {'residual':False,'art':False,'se':True},
            {'residual':False,'art':True,'se':True},
            {'residual':True,'art':True,'se':True},
            {'experimental_decoder':False}
            ]
    config_list = diff_to_conf(base,exp_diff_list,exp_group_diff)
    return config_list


def gen_decoders1907C_configs(base):
#    exp_group_name = 'decoders1907B_debug'
    exp_group_name = 'decoders1907C_maps1800'
    exp_group_diff = {
            'name':os.path.join(exp_group_name,'exp'),
            'experimental_decoder':True,
            'decoder_exp_setting':'C',
            'cv_k_todo':1
            }
    exp_diff_list = [
            {'residual':False,'art':False,'se':False},
            {'residual':True,'art':False,'se':False},
            {'residual':False,'art':True,'se':False},
            {'residual':False,'art':False,'se':True},
            {'residual':False,'art':True,'se':True},
            {'residual':True,'art':True,'se':True},
            {'experimental_decoder':False}
            ]
    config_list = diff_to_conf(base,exp_diff_list,exp_group_diff)
    return config_list


def gen_table1_configs(base):
    exp_group_name = 'exp_group_table1'
    exp_group_diff = {'name':os.path.join(exp_group_name,'exp')}
    exp_diff_list = [
            {'dataset':'original_all',
                'v_flip':0,'h_flip':0,'rot':0,'contrast':0,'brightness':0},
            {'dataset':'maps1800_all',
                'v_flip':0,'h_flip':0,'rot':0,'contrast':0,'brightness':0},
            {'dataset':'original_all'},
            {'exp_group_baseline':True}
            ]

    config_list = diff_to_conf(base,exp_diff_list,exp_group_diff)
    return config_list


def gen_table2_configs(base):
    exp_group_name = 'exp_group_table2'
    exp_group_diff = {'name':os.path.join(exp_group_name,'exp')}
    exp_diff_list = [
            {'backbone':'resnet101','se':False},
            {'pretrained':False},
            {'predict_distance':False,'predict_contour':False},
            {'predict_distance':True,'predict_contour':False},
            {'predict_distance':False,'predict_contour':True},
            {'n_ensemble':0},
            {'postprocess':'none'},

            {'exp_group_baseline':True}
            ]
    config_list = diff_to_conf(base,exp_diff_list,exp_group_diff)
    return config_list


def gen_opt_lw_configs(base):
    exp_group_name = 'exp_group_opt_lw'
    exp_group_diff = {'name':os.path.join(exp_group_name,'exp')}
    exp_diff_list = [
            {'loss_weights':[1,1,1]},

            {'loss_weights':[2,1,1]},
            {'loss_weights':[1,2,1]},
            {'loss_weights':[1,1,2]},

            {'loss_weights':[2,2,1]},
            {'loss_weights':[2,1,2]},
            {'loss_weights':[1,2,2]},
            
            {'loss_weights':[4,1,1]},
            {'loss_weights':[1,4,1]},
            {'loss_weights':[1,1,4]},

            {'loss_weights':[4,4,1]},
            {'loss_weights':[4,1,4]},
            {'loss_weights':[1,4,4]},

            {'exp_group_baseline':True}
            ]
    config_list = diff_to_conf(base,exp_diff_list,exp_group_diff)
    return config_list


if __name__=="__main__":
    #Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--exp_group',default=None)
    args = parser.parse_args()
    argsdict = vars(args)
    print(argsdict)


    #########################################################################
    ### MANUAL CONFIGURATION of baseline and experiments to generate here ###
    best_model_path = os.path.join(*[project_dir,"baseline_configs",
        "best_model_1807.json"])
    best_model = json.loads(open(best_model_path,'r').read())

    #gen_x_configs() functions take in baselines and generate a list of dicts
    #corresponding to the json configs to be run in an experiment
    if argsdict['exp_group'] == 'table1':
        config_list = gen_table1_configs(good_model)
    elif argsdict['exp_group'] == 'table2':
        config_list = gen_table2_configs(best_model)
    elif argsdict['exp_group'] == 'opt_lw':
        config_list = gen_opt_lw_configs(good_model)

    elif argsdict['exp_group'] == '1907A_debug':
        config_list = gen_decoders1907A_configs(best_model)
    elif argsdict['exp_group'] == '1907A':
        config_list = gen_decoders1907A_configs(best_model)
    elif argsdict['exp_group'] == '1907B_debug':
        config_list = gen_decoders1907B_configs(best_model)
    elif argsdict['exp_group'] == '1907B':
        config_list = gen_decoders1907B_configs(best_model)
    elif argsdict['exp_group'] == '1907C':
        config_list = gen_decoders1907C_configs(best_model)
    elif argsdict['exp_group'] == '1907D':
        config_list = gen_decoders1907D_configs(best_model)
    #########################################################################


    #Delete experiment config folder with same name if exists,
    #and make new folder
    exp_group_config_path = os.path.join(project_dir,
            'config_'+argsdict['exp_group'])
    if os.path.exists(exp_group_config_path):
        shutil.rmtree(exp_group_config_path)
    os.makedirs(exp_group_config_path)

    #Dump jsons to config folder
    for i in range(len(config_list)):
        out_file_path = os.path.join(exp_group_config_path,
                "exp"+str(i).zfill(3)+".json")
        out_file = open( out_file_path, "w") 
        json.dump(config_list[i], out_file, indent = 6) 
        out_file.close()
