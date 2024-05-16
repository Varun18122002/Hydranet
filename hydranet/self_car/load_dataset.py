import os
from PIL import Image
import numpy as np
import torch

class HydraNetDataLoader():
    
    #defining the initialization of datasets
    def __init__(self, data_file , splits, dataset_type ,transform = None , splitfor = None):
        self.data_file =  data_file #path of the data
        self.splits = splits #split path as train , test and val
        self.transform = transform  
        self.dataset_type = dataset_type
        self.splitfor = splitfor
        
        #Loading Files for data and annotations 
        if dataset_type == "cityscapes":
            self.image_paths = self._load_paths(os.path.join(data_file,"maindata/data",splits))
            self.image_color_seg= self._load_annotations(os.path.join(data_file,"annotation/ann",splits))

        if dataset_type == "nyu_depth_v2" and splitfor == "train" :
            self.image_paths , self.image_depths = self._load_images(os.path.join(data_file,"nyu_data/data/nyu2_train"))

        if dataset_type == "nyu_depth_v2" and splitfor == "test" :
            self.image_paths , self.image_depths = self._load_images_test(os.path.join(data_file,"nyu_data/data/nyu2_test"))
            
       	if dataset_type == "nyu_depth_v2" and splitfor == "val" :
            self.image_paths , self.image_depths = self._load_images(os.path.join(data_file,"nyu_data/data/nyu2_val"))
            
        
    def _load_images(self, directory):
        images = []
        depths = []
        for root , dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    images.append(os.path.join(root,file))
                elif file.endswith('.png'):
                    depths.append(os.path.join(root,file))
        images = sorted(images)
        depths = sorted(depths)
        

        return images , depths
    
    def _load_images_test(self, directory):
        images = []
        depths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('colors.jpg') or file.endswith('colors.png'):
                    images.append(os.path.join(root, file))
                elif file.endswith('depth.png'):
                    depths.append(os.path.join(root, file))
        images = sorted(images)
        depths = sorted(depths)
        
        return images, depths
    
    def _load_image_as_tensor(self, image_path):
        image = np.array(Image.open(image_path).convert("RGB"))
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and adjust dimensions
        return image_tensor


    def _load_depths(directory):
        depths = []
        
        return depths
                
    # length of image dataset
    def __len__(self):
        return len(self.image_paths)     

        
    def _load_paths(self, directory):
        images = []
        for root, dirs , files in os.walk(directory):
            for file in files:
                images.append(os.path.join(root,file))
        images = sorted(images)
        
        return images
    
    def _load_annotations(self, annotation_dir):
        annotations = []
        colors = []
        for root, dirs, files in os.walk(annotation_dir):
            for file in files:
                if file.endswith('.json'):
                    annotations.append(os.path.join(root,file))
                elif file.endswith('color.png'):
                    colors.append(os.path.join(root,file))
        colors = sorted(colors)
        
        return colors
    
