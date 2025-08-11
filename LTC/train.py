import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import itertools
import matplotlib.pyplot as plt
from colorhash import ColorHash

def train_clusters(decoder, optimizer, kmeans, bias=False):
    losses = []
    epochs = 10000
    pbar = tqdm.tqdm(range(epochs), total=epochs)
    x = torch.from_numpy(kmeans.cluster_centers_)
    input_cw = torch.tensor(list(itertools.product([-1, 1], repeat=k))).float()
    print(input_cw.shape, x.shape)
    for epoch in pbar:
        xhat = decoder(input_cw)
        loss = torch.linalg.norm(x-xhat)**2 
        losses.append(loss.item()/ (x.shape[0]*x.shape[1]))
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Loss={losses[-1]}')
    return losses

def train(encoder, decoder, act, optimizer, dset, batch_size, device, epochs=20):
    losses = []

    pbar = tqdm.tqdm(range(epochs), total=epochs)
    for epoch in pbar:
        dset2 = dset[torch.randperm(1000*batch_size)]
        for i in range(1000):
            # x = torch.randn((batch_size, d))
            x = dset2[i*batch_size:i*batch_size+batch_size]
            x = x.to(device)
            # print(x.shape)
            
            optimizer.zero_grad()
            z = act(encoder(x))
            # print(z)
            xhat = decoder(z)
            loss = torch.linalg.norm(x-xhat)**2 
            losses.append(loss.item()/ (x.shape[0]*x.shape[1]))
            loss.backward()
            optimizer.step()
        pbar.set_description(f'Loss={np.mean(losses[-100:])}')
    return losses

def trainNTC_old(model, lam, optimizer, aux_optimizer, dset, batch_size, device, epochs=20):
    DD = []
    RR = []

    pbar = tqdm.tqdm(range(epochs), total=epochs)
    for epoch in pbar:
        dset2 = dset[torch.randperm(1000*batch_size)]
        for i in range(1000):
            # x = torch.randn((batch_size, d))
            x = dset2[i*batch_size:i*batch_size+batch_size]
            x = x.to(device)
            # print(x.shape)
            
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            xhat, y_lik = model(x)
            D = torch.linalg.norm(x-xhat)**2  / (x.shape[0]*x.shape[1])
            R = torch.mean(-torch.log2(y_lik))
            loss = R + lam * D
            DD.append(D.item())
            RR.append(R.item())
            loss.backward()
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
        pbar.set_description(f'R={np.mean(RR[-100:]):.4f}, D={np.mean(DD[-100:]):.4f}')
    return RR, DD

def trainNTC(model, lam, optimizer, aux_optimizer, loader, eval_loader, device, args, dist_loss=F.mse_loss):
    DD = []
    RR = []

    for epoch in range(args.epochs):
        # print(f"epoch={epoch}", len(loader))
        # dset2 = dset[torch.randperm(1000*batch_size)]
        pbar = tqdm.tqdm(loader, total=len(loader), dynamic_ncols=True)
        for i, x in enumerate(pbar):
            x = x[0].to(device)
            
            optimizer.zero_grad()
            xhat, y_lik = model(x)
            # D = torch.mean((x-xhat)**2)
            D = dist_loss(x, xhat)
            if args.loglik:
                R = torch.sum(-y_lik * np.log2(np.e)) / (x.shape[0]*args.n)
            else:
                R = torch.sum(-torch.log2(y_lik)) / (x.shape[0]*args.n)
            
            loss = R + lam * D
            DD.append(D.item())
            RR.append(R.item())
            loss.backward()
            if args.clip_grad_norm > 0.:
                # print(clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            if aux_optimizer is not None:
                aux_optimizer.zero_grad()
                aux_loss = model.aux_loss()
                aux_loss.backward()
                aux_optimizer.step()
            else:
                aux_loss = 0
            total_norm = 0
            # for p in model.g_a.parameters():
            #     param_norm = p.grad.detach().data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"grad_norm={total_norm}")
            pbar.set_description(f'Epoch={epoch}, R={np.mean(RR[-20:]):.4}, D={np.mean(DD[-20:]):.4}')

        if epoch % args.eval_freq == 0 and epoch > 0:
            r, d = evalNTC(model, eval_loader, device, d=args.n,
                                                   dist_loss=dist_loss,
                                                   N_integral=args.N_integral_eval)
            print(f"Eval rate={r:.4}, dist={d:.4}")
    return RR, DD

# def evalNTC(model, device='cpu', n=20000, d=2):
#     """
#     TODO: write function that evals true rate (and distortion). 
#     Can be called after training a model for a rate sweep
#     """
#     bsize = 500
#     x = torch.randn((n, d), device=device)
#     xq = []
#     lik = []
#     y = []
#     y_hat = []
#     for i in tqdm.trange(len(x) // bsize):
#         xq1, lik1, y1, y_hat1 = model.eval(x[i*bsize:i*bsize+bsize])
#         xq.append(xq1)
#         lik.append(lik1)
#         y.append(y1)
#         y_hat.append(y_hat1)
#     x_hat = torch.cat(xq).detach().cpu()
#     lik = torch.cat(lik).detach().cpu()
#     y = torch.cat(y).detach().cpu()
#     y_hat = torch.cat(y_hat).detach().cpu()
#     x = x.cpu()
#     C2 = torch.unique(torch.cat((x_hat, y_hat), dim=1), dim=0)
#     C, C_latent = C2[:,:d], C2[:,d:]
#     latent_dist = []
#     for i in tqdm.trange(C_latent.shape[0]):
#         M = np.all(y_hat.numpy() == C_latent[i, :].numpy(), axis=1)
#         latent_dist.append(np.sum(M) / len(M)) # append frequency
#     # print(C, C_latent)
#     rate = -np.sum(latent_dist * np.log2(latent_dist)) / d
#     rate_fixed = np.log2(C_latent.shape[0]) / d
#     distortion = torch.mean((x-x_hat)**2).item()
#     return rate, distortion, rate_fixed

def evalNTC_true_ent(model, loader, device='cpu', n=20000, d=2):
    """
    TODO: write function that evals true rate (and distortion). 
    Can be called after training a model for a rate sweep
    """
    bsize = 200
    x = loader.dataset.tensors[0]
    # x = torch.randn(n, d)
    # loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=bsize)
    xq = []
    lik = []
    y = []
    y_hat = []
    # for i in tqdm.trange(len(x) // bsize, dynamic_ncols=True):
    pbar = tqdm.tqdm(loader, total=len(loader))
    for _, x_batch in enumerate(pbar):
        x_batch = x_batch[0].to(device)
        xq1, lik1, y1, y_hat1 = model.eval(x_batch)
        xq.append(xq1.detach().cpu())
        lik.append(lik1.detach().cpu())
        y.append(y1.detach().cpu())
        y_hat.append(y_hat1.detach().cpu())
    x_hat = torch.cat(xq).detach().cpu()
    lik = torch.cat(lik).detach().cpu()
    y = torch.cat(y).detach().cpu()
    y_hat = torch.cat(y_hat).detach().cpu()
    x = x.cpu()
    C_latent, counts = torch.unique(torch.cat((x_hat.flatten(start_dim=1), y_hat), dim=1), dim=0, return_counts=True)
    latent_dist = counts / y_hat.shape[0]
    # print(C, C_latent)
    rate = -torch.sum(latent_dist * torch.log2(latent_dist)).item() / d
    # rate_fixed = np.log2(C_latent.shape[0]) / d
    rate_ub = torch.sum(-torch.log2(lik)).item() / (n * d)
    distortion = torch.mean((x-x_hat)**2).item()
    return rate, distortion, rate_ub

def evalNTC(model, loader, device='cpu', d=2, dist_loss=F.mse_loss, N_integral=2048):

    # for i in tqdm.trange(len(x) // bsize, dynamic_ncols=True):
    rate = 0
    distortion = 0
    pbar = tqdm.tqdm(loader, total=len(loader), dynamic_ncols=True)
    for _, x_batch in enumerate(pbar):
        x_batch = x_batch[0].to(device)
        xq1, lik1 = model.eval(x_batch, N=N_integral)
        rate += torch.sum(-torch.log2(lik1)).item() / (x_batch.shape[0] * d) # bits per sample
        # distortion += torch.mean((x_batch-xq1)**2).item() # 
        distortion += dist_loss(x_batch, xq1).item()

    return rate / len(loader), distortion / len(loader)

def evalNTC_fixed_rate(model, loader, device='cpu', d=2, dist_loss=F.mse_loss, N_integral=2048):

    # for i in tqdm.trange(len(x) // bsize, dynamic_ncols=True):
    rate = 0
    rate_fixed =0
    distortion = 0
    pbar = tqdm.tqdm(loader, total=len(loader), dynamic_ncols=True)
    for _, x_batch in enumerate(pbar):
        x_batch = x_batch[0].to(device)
        xq1, lik1, y, y_hat = model.eval(x_batch, N=N_integral, return_y=True)
        rate += torch.sum(-torch.log2(lik1)).item() / (x_batch.shape[0] * d) # bits per sample
        rate_fixed += np.log2(torch.unique(y_hat, dim=0).shape[0]) / d
        # distortion += torch.mean((x_batch-xq1)**2).item() # 
        distortion += dist_loss(x_batch, xq1).item()

    return rate / len(loader), rate_fixed / len(loader), distortion / len(loader)


def eval(encoder, decoder, act):
    with torch.no_grad():
        n = 1000000
        x = torch.randn((n, d))
        z = act(encoder(x))
        # print(z)
        xhat = decoder(z)
        loss = torch.linalg.norm(x-xhat)**2 / (n * d)
    return loss.item()


def plot_quantizer(x, x_hat, y, y_hat):
    C2 = torch.unique(torch.cat((x_hat, y_hat), dim=1), dim=0)
    C, C_latent = C2[:,:2], C2[:,2:]

    C_colors = []
    Ms = []
    latent_dist = []
    # plt.figure(figsize=(12,8))
    # plt.subplot(1, 2, 1)
    # plt.figure(1, figsize=(6,6))
    for i in range(C_latent.shape[0]):
        M = np.all(y_hat.numpy() == C_latent[i, :].numpy(), axis=1)
        Ms.append(M)
        # print(np.sum(M) / len(M))
        latent_dist.append(np.sum(M) / len(M)) # append frequency
        # color_i = np.random.rand(3) * 0.5 + 0.5
        color_i = (np.array(ColorHash(C_latent[i,:]).rgb, dtype=float) / 255) * 0.75 + 0.25
        C_colors.append(color_i.tolist())
    #     plt.scatter(y[M,0], y[M,1], s=2, alpha=0.75, color=color_i)
    # plt.scatter(C_latent[:, 0], C_latent[:, 1], marker='o', c=np.array(C_colors), edgecolors='black', s=20, label=r'$\hat{Y}$')
    # ax = plt.gca()
    # ax.set_aspect('equal')
    # plt.grid(visible=True)
    # plt.title('Latent Space')
    # plt.legend()
    # plt.show()

    # plt.figure(2, figsize=(6,6))
    plots = []
    for i in range(C.shape[0]):
        M = Ms[i]
        plot1 = plt.scatter(x[M,0], x[M,1], s=2, alpha=0.75, color=C_colors[i], animated=True)
        plots.append(plot1)
    plot2 = plt.scatter(C[:, 0], C[:, 1], marker='o', c=np.array(C_colors), edgecolors='black', s=20, animated=True)
    plots.append(plot2)
    # plt.legend()
    # plt.show()
    return plots

def plot_quantizer2D(x, x_hat):
    C = torch.unique(x_hat, dim=0)

    C_colors = []
    Ms = []
    latent_dist = []

    for i in range(C.shape[0]):
        M = np.all(x_hat.numpy() == C[i, :].numpy(), axis=1)
        Ms.append(M)
        # print(np.sum(M) / len(M))
        latent_dist.append(np.sum(M) / len(M)) # append frequency
        # color_i = np.random.rand(3) * 0.5 + 0.5
        color_i = (np.array(ColorHash(C[i,:]).rgb, dtype=float) / 255) * 0.75 + 0.25
        C_colors.append(color_i.tolist())

    for i in range(C.shape[0]):
        M = Ms[i]
        plt.scatter(x[M,0], x[M,1], s=2, alpha=0.75, color=C_colors[i])
    plt.scatter(C[:, 0], C[:, 1], marker='o', c=np.array(C_colors), edgecolors='black', s=20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.grid(visible=True)
    plt.show()

# import meshplot
# def plot_quantizer3D(x, x_hat):
#     m = torch.min(x, dim=0)[0].numpy()
#     ma = torch.max(x, dim=0)[0].numpy()    

#     # Corners of the bounding box
#     v_box = np.array([[m[0], m[1], m[2]], [ma[0], m[1], m[2]], [ma[0], ma[1], m[2]], [m[0], ma[1], m[2]],
#                     [m[0], m[1], ma[2]], [ma[0], m[1], ma[2]], [ma[0], ma[1], ma[2]], [m[0], ma[1], ma[2]]])

#     # Edges of the bounding box
#     f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], 
#                     [7, 4], [0, 4], [1, 5], [2, 6], [7, 3]], dtype=int)
#     C = torch.unique(x_hat, dim=0)

#     C_colors = []
#     Ms = []
#     latent_dist = []
#     for i in range(C.shape[0]):
#         M = np.all(x_hat.numpy() == C[i, :].numpy(), axis=1)
#         Ms.append(M)
#         # print(np.sum(M) / len(M))
#         latent_dist.append(np.sum(M) / len(M)) # append frequency
#         # color_i = np.random.rand(3) * 0.5 + 0.5
#         color_i = (np.array(ColorHash(C[i,:]).rgb, dtype=float) / 255) * 0.75 + 0.25
#         C_colors.append(color_i.tolist())
#     p = meshplot.plot(C.numpy(), c=np.array(C_colors), shading={"point_size": 1, "marker":'o', "edgecolors":'black'}, return_plot=True)
#     for i in range(C.shape[0]):
#         M = Ms[i]
#         c = np.tile(np.array(C_colors[i]), (M.sum(), 1))
#         p.add_points(x[M].numpy(), c=c, shading={"point_size": 0.12})
#     p.add_edges(v_box, f_box, shading={"line_color": "red"})
#     p.add_points(v_box, shading={"point_color": "green"})


def trainNTC_with_plot(model, lam, optimizer, aux_optimizer, loader, device, epochs=20, update_freq=10):
    DD = []
    RR = []
    from IPython import display
    import matplotlib.animation as animation

    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.grid(visible=True)
    plt.title('Original Space')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    x_test = torch.randn((10000,2), device=device)
    frames = []
    for epoch in range(epochs):
        # dset2 = dset[torch.randperm(1000*batch_size)]
        pbar = tqdm.tqdm(loader, total=len(loader))
        for i, x in enumerate(pbar):
            # x = torch.randn((batch_size, d))
            # x = dset2[i*batch_size:i*batch_size+batch_size]
            x = x[0].to(device)
            # print(x.shape)
            
            optimizer.zero_grad()
            xhat, y_lik = model(x)

            if i % update_freq == 0:
                x_hat, lik, y, y_hat = model.eval(x_test)
                # plt.clf()
                plots = plot_quantizer(x_test.cpu(), x_hat.cpu(), y.cpu(), y_hat.cpu())
                frames.append(plots)
                # display.clear_output(wait=True)
                # display.display(plt.gcf())
            # print(y_lik.max().item(), y_lik.min().item())
            # print(x.shape, xhat.shape)
            D = torch.linalg.norm(x-xhat)**2  / (x.shape[0]*x.shape[1])
            R = torch.sum(-torch.log2(y_lik)) / (x.shape[0]*x.shape[1])
            # R = torch.mean(-torch.log2(y_lik))
            loss = R + lam * D
            DD.append(D.item())
            RR.append(R.item())
            loss.backward()
            optimizer.step()

            if aux_optimizer is not None:
                aux_optimizer.zero_grad()
                aux_loss = model.aux_loss()
                aux_loss.backward()
                aux_optimizer.step()
            else:
                aux_loss = 0
            pbar.set_description(f'Epoch={epoch}, R={np.mean(RR[-min(100, len(RR)):]):.4f}, D={np.mean(DD[-min(100, len(DD)):]):.4f}, aux_loss={aux_loss:.4f}')
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
    ani.save(f'plots/{model.lattice_name}_{model.transform_name}_lam{lam}.mp4')
    plt.show()
    return RR, DD