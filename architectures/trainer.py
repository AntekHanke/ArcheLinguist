import torch.nn
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import utilfunctions
import logging

logger = logging.getLogger(__name__)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    t = torch.ones(train_loader.batch_size).to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, ims = model(data)
        pairwise_diffs = utilfunctions.get_pairwise_differences(ims)
        toosimloss = -torch.mean(torch.square(pairwise_diffs))*10

        output = output.reshape(target.size())

        reconstruct = model.loss(output, target, t)
        inkused = torch.mean(ims)*10
        #loss = reconstruct/100 + toosimloss + inkused/2000

        loss=toosimloss + inkused + reconstruct

        logger.info("Loss: ", loss.item(), "Of which: Inkused=", inkused.item(), " reconstruct=", reconstruct.item(), " toosim=", toosimloss.item())
        loss.backward()
        optimizer.step()


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def visualize(model, device, test_loader, word_list):
    model.eval()
    data = test_loader.dataset.embeddings
    n = 5
    r = 2
    with torch.no_grad():
        fig, axs = plt.subplots(r,n)

        embs = torch.from_numpy(np.asarray(data.iloc[:n*r,0].values.tolist())).to(device)
        imss = model.getImage(embs).cpu()
        meanim = imss.mean(dim=0)*0
        for i in range(n):
            for rs in range(r):
                word = data.iloc[i+rs*n][1]
                ims = axs[rs,i].imshow((imss[i+rs*n,:,:,:]-meanim).permute(1,2,0))#, vmin=0, vmax=1)
                axs[rs,i].set_title(word)
        fig.colorbar(ims, ax = axs[:])
        plt.show()


from data.dataset import EmbeddingsDataset
from autoencoder import Encoder, Decoder, AutoEncoder
from perturber import Perturber
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='ArcheLinguist Train')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=304, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.999, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset = EmbeddingsDataset(".\\data\\train10.pkl", duplicate=1)
    encoder = Encoder()
    decoder = Decoder(embedding=dataset.embedding)
    perturber = Perturber()
    model = AutoEncoder(encoder, perturber, decoder).to(device)

    train_loader =  torch.utils.data.DataLoader(dataset, **train_kwargs)

    # model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        visualize(model, device, train_loader, ['person', 'year', 'way', 'day', 'thing', 'man'])
        train(args, model, device, train_loader, optimizer, epoch)

        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "autoencoder.pt")


if __name__ == '__main__':
    main()