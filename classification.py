import glob
import os
import argparse
import shutil

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--classification", type=str, required=True,
	help="which classification you need")
    
args = vars(ap.parse_args())

classification_path = os.path.join(args["classification"])

print('INFO:This script is aimed to classify the images and text files')

path_file_name=glob.glob('images/segmentation/train/*_json')
file_num = len(path_file_name)
print('INFO:There are ' + str(file_num) + ' folders need to be dealt')

PATH_TO_CLASSIFICATION_DIR = "images/segmentation/train"
out_dir = [os.path.join(PATH_TO_CLASSIFICATION_DIR, classification_path.split('.')[0])]
out_dir = out_dir[0]
print('INFO:' + classification_path + ' will be dealt')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
print('INFO:' + 'your destination folder is ' + out_dir)
for i in range(1,file_num + 1):
    source_file = out_dir + '/' + classification_path
    destination_file = out_dir + '/{}'.format(i) + '.' + classification_path.split('.')[-1]
    temp_file = out_dir + '/' + classification_path
    classification_file_name = os.path.join(PATH_TO_CLASSIFICATION_DIR, '{}_json'.format(i), classification_path)
    print('INFO:' + classification_file_name)
    if not os.path.exists(destination_file):
        shutil.copy(classification_file_name,out_dir)
        os.rename(source_file, destination_file)

print('INFO:finished')
