import base64
import json
import os
import os.path as osp

import PIL.Image
import yaml

from labelme.logger import logger
from labelme import utils
import glob


def main():
    print('This script is aimed to demonstrate how to convert the'
          'JSON file to a single image dataset, and not to handle'
          'multiple JSON files to generate a real-use dataset.')
   
    path_file_name=glob.glob('images/segmentation/train/*.json')
    file_num = len(path_file_name)
    print('INFO:There are ' + str(file_num) + ' json files')
    file_name = [i for i in range(file_num)]
    for i in range(file_num):
        file_name[i] = path_file_name[i].split('\\')[-1]
        print('INFO:' + file_name[i] + ' is dealt')
        data = json.load(open(path_file_name[i]))
        imageData = data.get('imageData')
        
        
        out_dir = osp.basename(path_file_name[i]).replace('.', '_')
        out_dir = osp.join(osp.dirname(path_file_name[i]), out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        if not imageData:
            imagePath = os.path.join(os.path.dirname(path_file_name[i]), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = utils.draw_label(lbl, img, label_names)

        PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
        utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        print('INFO:info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('INFO:Saved to: {}'.format('images/segmentation/test/' + file_name[i]))

    print('INFO:finished!')

if __name__ == '__main__':
    main()
