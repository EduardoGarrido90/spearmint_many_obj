import os
import json
import shutil
import sys
from spearmint.main import main as spearmint_main 
from spearmint.cleanup import cleanup 
def main():
    src = 'examples/fifth_expt/'
    thresholds = [0.05]
    use_distance_from_iteration = [20]
    #thresholds = [0.05, 0.1, 0.2]
    #use_distance_from_iteration = [5, 10, 15]

    old_cwd = os.getcwd()
    for threshold in thresholds:
        for iteration in use_distance_from_iteration:
            dir_name = 'expt5_'+ "{:.2f}".format(threshold)[2:] + '_' + str(iteration) + '/'
            
            create_expt_dir(dir_name, src)
            
            os.chdir(dir_name)
            
            update_config(threshold, iteration)
            
            print('\n\nExecuting spearmint main for ' + dir_name); spearmint_main('.')
            
            hypervolumes()

            print('Cleaning up experiment\n\n'); cleanup('.')
            
            os.chdir(old_cwd)
            


def create_expt_dir(dir_name , src):
    # copy of the experiment dir
    if not os.path.exists(dir_name):     
        print('Creating dir '+ dir_name)
        shutil.copytree(src, dir_name, ignore = ignore_pyc_log_files)
    return
def update_config(threshold, iteration):
    # update config.json with threshold and iteration
    with open('config.json', 'r') as fd:
        config = json.load(fd)
    config['apply__distance'] = True
    config['threshold'] = threshold
    config['use_distance_from_iteration'] = iteration
    config['experiment-name'] = 'branin-'+ str(threshold)[2:] + '-' + str(iteration)
    with open('config.json', 'w') as fd:
        json.dump(config, fd, indent = 4)
    return
def hypervolumes():
    sys.path.append(os.path.realpath('.'))
    module = __import__('generate_hypervolumes')
    print('Executing generate_hypervolumes')                       
    module.main('.')
    return
    

def ignore_pyc_log_files(dirname, filenames):
    return list(filter(pyc_log_files, filenames))

def pyc_log_files(filename):
    return bool(filename.startswith('db') or filename.startswith('log') or filename.endswith('pyc'))

if __name__ == "__main__":
    main()

