import json
import os
import argparse


#official
def gen_table1(base):
    exp_group_name = 'exp_group_table1'
    exp_group_diff = {'name':os.path.join(exp_group_name,'exp')}
    exp_diff_list = [
            {'dataset':'original_all','preprocess':False},
            {'dataset':'original_all','preprocess':True},
            {'dataset':'maps1800_all','preprocess':False},
            {'dataset':'maps1800_all','preprocess':True}
            ]
    config_list = []
    for exp_diff in exp_diff_list:
        config_tmp = base.copy()
        for k,v in exp_group_diff.items():
            config_tmp[k] = v
        for k,v in exp_diff.items():
            config_tmp[k] = v
        config_list.append(config_tmp)
    return(config_list)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--exp_group',default=None)
    args = parser.parse_args()
    argsdict = vars(args)
    print(argsdict)

    project_dir = ""
    internal_baseline_path = os.path.join(*[project_dir,"config_archive","internal_baseline.json"])
    internal_baseline = json.loads(open(internal_baseline_path,'r').read())
    best_model_path = os.path.join(*[project_dir,"config_archive","best_model_0707.json"])
    best_model = json.loads(open(best_model_path,'r').read())
#    intermediate_baseline_path = os.path.join(*[project_dir,"config_archive","intermediate_baseline.json"])
#    intermediate_baseline = json.loads(open(int_base_path,'r').read())
#    intermediate_baseline_path = os.path.join(*[project_dir,"config_archive","intermediate_baseline.json"])
#    intermediate_baseline = json.loads(open(int_base_path,'r').read())


    if argsdict['exp_group'] == 'table1':
        config_list = gen_table1(best_model)

    exp_group_config_path = os.path.join(project_dir,'config_'+argsdict['exp_group'])
    if os.path.exists(exp_group_config_path):
        shutil.rmtree(exp_group_config_path)
    os.makedirs(exp_group_config_path)
    for i in range(len(config_list)):
        out_file_path = os.path.join(exp_group_config_path,"exp"+str(i)+".json")
        out_file = open( out_file_path, "w") 
        json.dump(config_list[i], out_file, indent = 6) 
        out_file.close() 





