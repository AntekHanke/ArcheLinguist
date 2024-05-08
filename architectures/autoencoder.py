import torch
import torch.nn as nn
import numpy as np

writing_dict = torch.tensor([
[
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
],
[
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
],
[
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
],
[
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0]
],
[
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
],
[
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0]
]

])

writing_to_use = torch.tile(writing_dict, (1,1,4,4))
#print(writing_to_use.size()) torch.Size([1, 6, 20, 20])

# Define the generator model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(300, 512),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 256),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.Tanh()
        )
        self.writing_to_use = writing_to_use.to(torch.device("cuda"))
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=3, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Using sigmoid activation for pixel values between 0 and 1
        )
        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x):
        # print("I have to forward x: ", x.size())
        x = self.fc(x)
        x = self.sigm(x*100)
        # print("After FC, x is now: ", x.size())
        x = x.view(-1, 6, 4, 4)  # Reshape to (batch_size, channels, height, width)
        x = x*100
        # print(x[0,:,:,:])
        x = self.softmax(x)
        # print(x[0, :, :, :])
        x = torch.repeat_interleave(x, 5, dim=3)
        x = torch.repeat_interleave(x, 5, dim=2)
        # print("Now after interleave x is: ", x.size())
        # print(x)
        x = x*self.writing_to_use
        # print("x after mult: ", x.size())
        x = x.sum(dim=1, keepdims=True)
        # print("x after sum along dim: ", x.size())
        #x = self.sigm(x)
        x = self.sigm(20*(x-torch.tensor(0.5)))
        #x = self.conv_transpose(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.embedding = embedding
        self.emb_vals = embedding.iloc[:,0].values
        self.np_vals = np.array(self.emb_vals.tolist())
        self.emb_tensor = torch.tensor(self.np_vals).to(torch.device("cuda")) #(words, 300)
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, self.embedding.shape[0])
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        # print("I have size in decoder: ", x.size())
        x = x.view(-1, 512)
        x = self.fc(x)
        x = self.softmax(x) #(batch, words)
        x = torch.einsum('bw,wv->bv', x, self.emb_tensor)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder, perturber, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.perturber = perturber
        self.decoder = decoder
        self.loss = torch.nn.CosineEmbeddingLoss()
        self.weights_init_normal()
        # self.bnorm = nn.BatchNorm1d(300)
        # self.test = nn.Linear(300, 300)

    def forward(self, x):
        # x = self.bnorm(x)
        ims = self.encoder(x)
        #inkused = torch.mean(x)
        x = self.perturber(ims)

        x = self.decoder(x)
        # x = self.test(x)
        return x, ims

    def getImage(self, x):
        im = self.encoder(x)
        return im

    def weights_init_normal(self):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = self.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = self.in_features
        # m.weight.data shoud be taken from a normal distribution
            self.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
            self.bias.data.fill_(0)
