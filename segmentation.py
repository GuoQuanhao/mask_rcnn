# coding=utf-8
import os
import glob

path_file_name=glob.glob('optional/*.json')
file_num = len(path_file_name)
file_name = [i for i in range(file_num)]
for i in range(file_num):
    file_name[i] = path_file_name[i].split('\\')[-1]
    print("INFO:" + file_name[i])
print("INFO:There are " + str(file_num) +" json files")
json_files = [ os.path.join('optional/', '{}.json'.format(i)) for i in range(1, file_num + 1) ]

for json_file in json_files:
    run = "labelme_json_to_dataset.exe %s" % (json_file)
    os.system(run)
    
print("INFO:finished")