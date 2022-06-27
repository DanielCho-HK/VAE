import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from vae import VAE
import os
import torch.nn.functional as F


def main():
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
    if not os.path.exists('./checkpoint/'):
        os.makedirs('./checkpoint/')

    mnist_train = datasets.MNIST('./mnist', True, transform=transforms.Compose(
      [transforms.ToTensor()]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('./mnist', False, transform=transforms.Compose(
        [transforms.ToTensor()]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    model = VAE().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter('./logs/')

    total_step = 1
    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            x = x.cuda()
            # batchsz = x.size(0)
            x_hat, mu, log_var = model(x)
            recon_loss = F.l1_loss(x_hat, x, size_average=False)
            kld = torch.sum(torch.exp(log_var) - (1 + log_var) + torch.pow(mu, 2))
            
            total_loss = recon_loss + kld
            writer.add_scalar('total_loss', total_loss.item(), total_step)
            writer.add_scalar('recon_loss', recon_loss.item(), total_step)
            writer.add_scalar('kld', kld.item(), total_step)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print('Epoch [%d/%d], Step [%d/%d], total_loss: %.4f, recon_loss: %.4f, kld: %.4f' %
                  (epoch + 1, 1000, batchidx + 1, len(mnist_train), total_loss.item(), recon_loss.item(), kld.item()))

            if total_step % 10000 == 0:
                with torch.no_grad():
                    x, _ = next(iter(mnist_test))
                    x = x.cuda()
                    x_hat, _, _ = model(x)
                    x = x.cpu().data
                    x_hat = x_hat.cpu().data
                    x = make_grid(x, 4, 0)
                    save_image(x, './results/' + 'in_{}.jpg'.format(total_step))
                    x_hat = make_grid(x_hat, 4, 0)
                    save_image(x_hat, './results/' + 'out_{}.jpg'.format(total_step))

            total_step += 1

        if epoch % 10 == 0:
            torch.save({'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'epoch': epoch, 
                        'total_step': total_step}, './checkpoint/' + 'gen_param{}.pth'.format(epoch))



if __name__ == '__main__':
    main()












