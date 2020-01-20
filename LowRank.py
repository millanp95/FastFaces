#This Script will:
#1. Load a model and for a given layer, maps the tensor to a matrix
#2. Compute de SVD of the Layer.
#3. Use the SVD to build the Low-Rank Approximation.
#4. Compute the Low-rank Factorization of the layer.
#5. Map it back to the network and make a prediction.


from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.shrinked_mb1 import create_shrinked_mobilenetv1_ssd, create_shrinked_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer
import cv2
import sys
import math
import torch
import numpy as np
import pandas as pd

#class_names = [name for name in open(label_path).readlines()]
class_names = ['BACKGROUND','Human head']
num_classes = len(class_names)

#Definition of the mapping Tau
def Tau(w):
    C = w.shape[1]
    N = w.shape[0]
    d = w.shape[2]

    W = np.zeros((C*d,N*d))

    for j_1 in range(C*d):
        for j_2 in range(N*d):
            i_1 = j_1 // d ; i_2=j_1 % d;
            i_4 = j_2 // d ; i_3=j_2 % d;
            W[j_1][j_2] = w[i_4,i_1,i_2,i_3]

    return W

#Definition of the inverse Mapping
def Tau_inv(W,K,d):

    #Get the dimensions of the kernels and build empy tensors.
    C = W.shape[0]//d; N = W.shape[1]//d;

    V = torch.empty((K,C,d,1))
    H = torch.empty((N,K,1,d))

    #Compute the SVD
    u, s, vh = np.linalg.svd(W, full_matrices=False)

    for c in range(C):
        for k in range(K):
            u_mat=np.resize(u[c*d:(c+1)*d,k],(d,1))
            V[k,c,:,:]=np.sqrt(s[k])*torch.from_numpy(u_mat)

    for n in range(N):
        for k in range(K):
            v_mat=np.resize(vh[k,N*d:(N+1)*d],(1,d))
            H[n,k,:,:]=np.sqrt(s[k])*torch.from_numpy(v_mat)

    return V,H

timer = Timer()

#pretrained Model
model_path = 'models/BaseModel.pth'
k = 4  #Rank

#Create the network using the architectures.
net = create_mobilenetv1_ssd(2, is_test=True)
net_copy = create_shrinked_mobilenetv1_ssd(2,4,is_test=True)

#Load the original model
net.load(model_path)

#Extract the parameters of the original model
params=net.state_dict()

print(params.keys())

for layer in net_copy.state_dict().keys():
    print(layer,net_copy.state_dict()[layer].shape)

#Select the layer in the original model for computing the Low-Rank Factorization:
layer='extras.0.2.weight'
tensor=params[layer]

#Map the tensor using Tau
W=Tau(tensor)

#Build the new filters
k = 4
V,H = Tau_inv(W,k,3)

#Update new layers shapes according to the factorization
params['extras.0.2.weight'] = V #torch.zeros([k,256,3,1])
bias=params['extras.0.2.bias']
bias+torch.zeros([512])
params['extras.0.2.bias']=torch.zeros([k])

#Add the new Layers to the parameters dict
params.update({'extras.0.4.weight': H,'extras.0.4.bias':bias})
#params.update({'extras.0.4.weight': torch.zeros([512,k,1,3]),'extras.0.4.bias':torch.zeros([512])})

#add the new network
net_copy.load_state_dict(params)

#Create the predictors based on the architecture:
predictor = create_shrinked_mobilenetv1_ssd_predictor(net_copy, candidate_size=200)
#predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

r = pd.read_csv('Data_model_TM1/open_images/sub-test-annotations-bbox.csv') #Change this line according to yor Dataset

for image_id, g in r.groupby('ImageID'):

    img=cv2.imread('Data_model_TM1/open_images/test/'+image_id+'.jpg')
    print(image_id)
    #print(img.shape)

    for row in g.itertuples():
        x1=int(row.XMin)
        y1=int(row.YMin)
        x2=int(row.XMax)
        y2=int(row.YMax)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Make the prediction using the new model
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.6)
    interval = timer.end()

    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        if (class_names[labels[i]]=='Human head' and not math.isnan(box[0]) and not math.isnan(box[1]) and not math.isnan(box[2]) and not math.isnan(box[3])):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(img, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)

    cv2.imshow('predictions',img)

    cv2.waitKey(0)


#Load the new model.
net_copy.load_state_dict(params)


#Print the parameters in the new model
for layer in net_copy.state_dict().keys():
    print(layer,net_copy.state_dict()[layer].shape)
