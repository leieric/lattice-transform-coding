import torch
import torch.nn.functional as F
# torch.set_default_dtype(torch.float32)
# torch.set_float32_matmul_precision('medium')
torch.set_num_threads(4)
# torch.autograd.set_detect_anomaly(True)
from LTC.layers import get_model
from LTC.train import trainNTC, evalNTC
from argparse import ArgumentParser
import LTC.data as data
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_loss(loss_func):
    if loss_func == 'L1':
        return F.l1_loss
    elif loss_func == "L2":
        return F.mse_loss
    else:
        raise Exception("invalid loss_func")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n', default=1, type=int) # blocklength
    parser.add_argument('--d', type=int) # input source dimension
    parser.add_argument('--dy', default=1, type=int) # latent source dimension
    parser.add_argument('--d_hidden', default=100, type=int) # hidden layer dimension
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n_train_samples', default=1000000, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--model_name', default='LatticeCompander', type=str)
    parser.add_argument('--data_name', default='Gaussian', type=str)
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--lattice_name', default='Hexagonal', type=str)
    parser.add_argument('--transform_name', default='LinearNoBias', type=str)
    parser.add_argument('--eb_name', default='FactorizedPrior', type=str)
    parser.add_argument('--lam_sweep', nargs='+', help='RD tradeoff', type=float, required=True)
    parser.add_argument('--num_eval_samples', default=200000, type=int)
    parser.add_argument('--N_integral', default=2048, type=int)
    parser.add_argument('--N_integral_eval', default=4096, type=int)
    parser.add_argument('--MC_method', default="standard", type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--loglik', action='store_true')
    parser.add_argument('--loss_func', default='L2', type=str)
    parser.add_argument('--clip_grad_norm', default=0., type=float)
    parser.add_argument('--eval_freq', default=10, type=int)
    args = parser.parse_args()

    # args.N_integral_eval = args.N_integral if args.N_integral_eval == 0 else args.N_integral_eval

    loader = data.get_loader(args, args.n_train_samples, train=True)
    eval_loader = data.get_loader(args, args.num_eval_samples, drop_last=True, train=False)
    # model = get_compander(args)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total params={pytorch_total_params}")

    rates_Lattice = []
    dists_Lattice = []
    rates_ub = []
    print(args.lam_sweep)
    for lam in args.lam_sweep:
        # model = LatticeCompander(args.n, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
        args.lam = lam
        model = get_model(args)
        if args.pretrained:
            save_dir = f'trained/{args.data_name}'
            saved = torch.load(f'{save_dir}/{args.model_name}_{args.lattice_name}_{args.transform_name}_{args.eb_name}_n{args.n}_d{args.d}_dy{args.dy}_Nint{args.N_integral}_lam{lam}.pt', map_location='cpu')
            model.load_state_dict(saved)
        model = model.to(device)
        parameters = set(p for name, p in model.named_parameters() if not name.endswith(".quantiles"))
        aux_parameters = set(p for name, p in model.named_parameters() if name.endswith(".quantiles"))
        # optimizer = torch.optim.Adam(parameters, lr=5e-4)
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        # aux_optimizer = torch.optim.Adam(aux_parameters, lr=5e-3)
        try:
            Rs, Ds = trainNTC(model, lam, optimizer, None, loader, eval_loader, device, args, dist_loss=get_loss(args.loss_func))
                                                                        
        except ValueError:
            print("NaNs")
            continue # skip to next lambda
            # Rs, Ds = trainNTC_with_plot(model, lam, optimizer, None, loader, device, epochs=2, update_freq=100)
        r, d = evalNTC(model, eval_loader, device, d=args.n,
                                                   dist_loss=get_loss(args.loss_func),
                                                   N_integral=args.N_integral_eval)
        print(f"Eval rate={r:.4}, dist={d:.4}")
        rates_Lattice.append(r)
        dists_Lattice.append(d)
        if args.save:
            save_dir = f'trained/{args.data_name}'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/{args.model_name}_{args.lattice_name}_{args.transform_name}_{args.eb_name}_n{args.n}_d{args.d}_dy{args.dy}_Nint{args.N_integral}_lam{lam}.pt')
    print(f'rates = {rates_Lattice}')
    print(f'dists = {dists_Lattice}')