
import os
import time
import cv2
import torch
import numpy as np
from PIL import Image

from torchvision import datasets
from torch.utils.data import DataLoader
from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import MTCNN


mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

dataset = datasets.ImageFolder('photos') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
load_data = torch.load('data.pt') 
embedding_list, name_list = load_data[0], load_data[1] 



def get_name(img_cropped_list, prob_list):
    ''' get the names of celebrities
    get the cropped img from mtcnn and return name of celebrities
    
    Args:
        img: image
        img_cropped_list: list of cropped images

    Returns: 
        name_list: name of the person
    '''
    if img_cropped_list is None:
        print('No face detected')
        return 
    
    celebrity_list = []
    for i, prob in enumerate(prob_list):
        if prob > 0.90:
            emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
            dist_list = [] # list of matched distances, minimum distance is used to identify the person
            
            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            min_dist = min(dist_list) # get minumum dist value
            if min_dist > 0.85:
                # print('No match found')
                name = 'nobody'
                celebrity_list.append(name)
            else:
                # print('match distance', min_dist)
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist\
                # print(f'we guess he/she is {name}')
                celebrity_list.append(name)
    return celebrity_list


if __name__ == '__main__':
    img = Image.open('cele_test/2.jpg') # image path
    img = np.array(img)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    name_list = get_name(img_cropped_list, prob_list)
    print(name_list)




# original_frame = frame.copy() # storing copy of frame before drawing on it

# if min_dist < 0.90:
#     frame = cv2.putText(frame, name+' '+str(min_dist), (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)

# new_img = cv2.rectangle(img, (box[0], box[1]) , (box[2], box[3]), (255, 0, 0), 2)





























# In[]
# cam = cv2.VideoCapture(0) 

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("fail to grab frame, try again")
#         break
        
#     img = Image.fromarray(frame)
#     img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    
#     if img_cropped_list is not None:
#         boxes, _ = mtcnn.detect(img)
                
#         for i, prob in enumerate(prob_list):
#             if prob>0.90:
#                 emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
#                 dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
#                 for idx, emb_db in enumerate(embedding_list):
#                     dist = torch.dist(emb, emb_db).item()
#                     dist_list.append(dist)

#                 min_dist = min(dist_list) # get minumum dist value
#                 min_dist_idx = dist_list.index(min_dist) # get minumum dist index
#                 name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
#                 box = boxes[i] 
                
#                 original_frame = frame.copy() # storing copy of frame before drawing on it
                
#                 if min_dist<0.90:
#                     frame = cv2.putText(frame, name+' '+str(min_dist), (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                
#                 frame = cv2.rectangle(frame, (box[0],box[1]) , (box[2],box[3]), (255,0,0), 2)

#     cv2.imshow("IMG", frame)
        
    
#     k = cv2.waitKey(1)
#     if k%256==27: # ESC
#         print('Esc pressed, closing...')
#         break
        
#     elif k%256==32: # space to save image
#         print('Enter your name :')
#         name = input()
        
#         # create directory if not exists
#         if not os.path.exists('photos/'+name):
#             os.mkdir('photos/'+name)
            
#         img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
#         cv2.imwrite(img_name, original_frame)
#         print(" saved: {}".format(img_name))
        
        
# cam.release()
# cv2.destroyAllWindows()
    


