import numpy as np
import torch
import os
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

k = 10

#orizoume to device gia ekpaideush sthn GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#8etoume to learning rate
lr = 0.00005

#pairnoume tis listes me ta onomata apo tous fakelous
train_image_names = os.listdir('./DataBase')
train_image_names.sort()
test_image_names = os.listdir('./test')
test_image_names.sort()

train_images = []
test_images = []

#prosthetoume se kathe lista ta dianusmata eikonwn
for img_name in train_image_names:
    train_images.append(cv2.imread('./DataBase/'+img_name, cv2.IMREAD_GRAYSCALE).flatten()) 
    
for img_name in test_image_names:
    test_images.append(cv2.imread('./test/'+img_name, cv2.IMREAD_GRAYSCALE).flatten()) 
    
train_images = np.array(train_images, dtype=np.double)
test_images = np.array(test_images, dtype=np.double)

#kanonikopoioume tis times twn eikonwn sto diasthma [0,1]
train_images /= 255
test_images /= 255

train_curve = [] 
test_curve = []

#orizoume thn arxitektonikh tou autoencoder
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(10000, 1000),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(1000, k)
        )
        self.decode = torch.nn.Sequential(
            torch.nn.Linear(k, 1000),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(1000, 10000),
        )        
    def encoder(self, x):
        return self.encode(x)
    def decoder(self, x):
        return self.decode(x)
    def forward(self, input):
        return self.decoder(self.encoder(input))

#orizoume th domh tou dataset mas
class myDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]        
        sample = {'data' : data}
        return sample

#dhmiourgoume ta train & test datasets
train_dataset = myDataset(train_images)
test_dataset = myDataset(test_images)

#dhmiourgoume tis train & test dataloaders me batch size 5 & 11 antistoixa
train_dataloader = DataLoader(train_dataset, batch_size = 5, shuffle=True, num_workers = 0)
test_dataloader = DataLoader(test_dataset, batch_size = 11, shuffle=False, num_workers = 0)

#arxikopoioume to diktyo mas
net = Net().double().to(device)

#orizoume ton optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

#orizoume to loss pou 8a xrhsimopoihsoume
mse_loss = torch.nn.MSELoss()

#ekpaideuoume gia 300 epoxes
for epoch in range(300):

    epoch_loss_train = 0

    for a, data in enumerate(train_dataloader):

        #pairnoume ena batch apo ta dedomena ekpaideushs      
        x = data['data'].double().to(device)

        #pairnoume thn eksodo tou diktuou
        x_rec = net(x)

        #upologizoume to sfalma anakataskeuhs tou batch        
        rec_loss = mse_loss(x_rec,x)

        #mhdenizoume ta gradients ths prohgoumenhs epoxhs, 
        #pairnoume ta nea gradients me vash to loss pou upologisame
        #& ananewnoume ta varh tou diktuou
        net.zero_grad()
        rec_loss.backward()
        optimizer.step()

        epoch_loss_train += rec_loss.item()

    #evaluation
    net.eval()

    epoch_loss_test = 0

    for b, data in enumerate(test_dataloader):
        #pairnoume ena batch apo ta dedomena elegxou
        x = data['data'].double().to(device)

        #pairnoume thn eksodo tou diktuou
        x_rec = net(x)

        #upologizoume to sfalma anakataskeuhs tou batch        
        rec_loss = mse_loss(x_rec,x)

        epoch_loss_test += rec_loss.item()

    net.train()
    
    #ektupwnoume to train & test loss
    print('Epoch : ',epoch)
    print('Train Loss :', epoch_loss_train/(a+1))
    print('Test Loss  :', epoch_loss_test/(b+1))
    
    train_curve.append(epoch_loss_train/(a+1))
    test_curve.append(epoch_loss_test/(b+1))

    #apo8hkeuoume tis kampules ekpaideushs & elegxou
    plt.figure(figsize=(15,10))
    plt.plot(train_curve)
    plt.plot(test_curve)
    plt.yscale('log')
    plt.title('Loss curves for autoencoder with latent size='+str(k))
    plt.legend(['Train Reconstruction Loss', 'Test Reconstruction Loss'])
    plt.savefig('./reconstruction_loss_'+str(k)+'.png')
    plt.close()


test_representations = []
train_representations = []

#prosthetoume se kathe lista ta dianusmata eikonwn
for img_name in train_image_names:
    img = cv2.imread('./DataBase/'+img_name, cv2.IMREAD_GRAYSCALE).flatten()
    tensor_img = torch.tensor(img,device=device).double().unsqueeze(0)
    latent_representation = net.encode(tensor_img)
    train_representations.append(latent_representation.detach().cpu().numpy())

#prosthetoume se kathe lista ta dianusmata eikonwn
for img_name in test_image_names:
    img = cv2.imread('./test/'+img_name, cv2.IMREAD_GRAYSCALE).flatten()
    tensor_img = torch.tensor(img,device=device).double().unsqueeze(0)
    latent_representation = net.encode(tensor_img)
    test_representations.append(latent_representation.detach().cpu().numpy())


success = 0
for i in range(len(test_representations)):
    dist = np.zeros((100))
    query = test_representations[i]
    for j in range(len(train_representations)):
        dist[j]=((query-train_representations[j])**2).mean()
    min_val = dist.min()
    if train_image_names[list(dist).index(min_val)] == test_image_names[i]:
        success += 1

print('Success rate for k='+str(k)+':',success/11*100,'%')
torch.save(net, './autoencoder_'+str(k)+'.pth')