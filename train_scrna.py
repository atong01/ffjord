import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob, uniform_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular

from diagnostics.viz_scrna import save_trajectory, trajectory_to_video, save_trajectory_density
from diagnostics.viz_toy import save_vectors

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument('--test', type=eval, default=False, choices=[True, False])
parser.add_argument('--full_data', type=eval, default=False, choices=[True, False])
parser.add_argument('--data', type=str, default='dummy')
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--alpha', type=float, default=0.0, help="loss weight parameter for growth model")
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--viz_batch_size', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")
parser.add_argument('--vecint', type=float, default=None, help="regularize direction")

parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

times = [args.time_length, 2*args.time_length]
import numpy as np
import atongtf.dataset as atd
from sklearn.preprocessing import StandardScaler

def load_data():
    data = atd.EB_Velocity_Dataset()
    labels = data.data['sample_labels']
    ixs = data.data['ixs']
    labels = labels[ixs]
    scaler = StandardScaler()
    scaler.fit(data.emb[ixs])
    transformed = scaler.transform(data.emb[ixs])
    return transformed, labels, scaler

def load_data_full():
    data = atd.EB_Velocity_Dataset()
    labels = data.data['sample_labels']
    scaler = StandardScaler()
    scaler.fit(data.emb)
    transformed = scaler.transform(data.emb)
    return transformed, labels, scaler

def load_circle_data():
    data = atd.Circle_Transition_Dataset(n=10000)
    labels = data.get_train_labels()
    full_data = data.get_train()
    transformed = full_data[:,:2]
    next_states = full_data[:,2:]
    directions = next_states - transformed
    return transformed, labels, np.concatenate([transformed, directions], axis=1)


#data, labels, data_and_directions = load_circle_data()
data, labels, scaler = load_data_full() if args.full_data else load_data()
timepoints = np.unique(labels)

#########
# SELECT TIMEPOINTS
#timepoints = timepoints[:2]

# Integration timepoints, where to evaluate the ODE
int_tps = (timepoints+1) * args.time_length

#########

def inf_sampler(arr, batch_size=None, noise=0.0):
    if batch_size is None: batch_size = args.batch_size
    ind = np.random.randint(len(arr), size=batch_size)
    samples = arr[ind]
    if noise > 0:
        samples += np.random.randn(*samples.shape) * noise
    return samples

def train_sampler(i):
    return inf_sampler(data[labels==i], noise=0.1)

def dir_train_sampler(i):
    return inf_sampler(data_and_directions[labels==i], noise=0.)

def val_sampler(i):
    return inf_sampler(data[labels==i], batch_size=args.test_batch_size)

def viz_sampler(i):
    return inf_sampler(data[labels==i], batch_size=args.viz_batch_size)

full_sampler = lambda: inf_sampler(data, 2000)

def scatter_timepoints():
    import matplotlib.pyplot as plt

    LOW = -4
    HIGH = 4
    fig, axes = plt.subplots(2,3, figsize=(20,10))
    axes = axes.flatten()
    titles = ['D00-03', 'D06-09', 'D12-15', 'D18-21', 'D24-27', 'Full']
    for i in range(5):
        ax = axes[i]
        dd = np.concatenate([train_sampler(i) for _ in range(10)])
        ax.hist2d(dd[:,0], dd[:,1], range=[[LOW,HIGH], [LOW,HIGH]], bins=100)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(titles[i])
    ax = axes[5]
    ax.hist2d(data[:,0], data[:,1], range=[[LOW,HIGH], [LOW,HIGH]], bins=100)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(titles[5])
    plt.savefig('scatter.png')
    plt.close()

#scatter_timepoints()

def get_transforms(model, integration_times):
    """
    Given a list of integration points,
    returns a function giving integration times
    """
    def sample_fn(z, logpz=None):
        int_list = [torch.tensor([it - args.time_length, it]).type(torch.float32).to(device) 
                for it in integration_times]
        if logpz is not None:
            # TODO this works right?
            for it in int_list:
                z, logpz = model(z, logpz, integration_times=it, reverse=True)
            return z, logpz
        else:
            for it in int_list:
                z = model(z, integration_times=it, reverse=True)
            return z

    def density_fn(x, logpx=None):
        int_list = [torch.tensor([it - args.time_length, it]).type(torch.float32).to(device)
                for it in integration_times[::-1]]
        if logpx is not None:
            for it in int_list:
                x, logpx = model(x, logpx, integration_times=it, reverse=False)
            return x, logpx
        else:
            for it in int_list:
                x =  model(x, integration_times=it, reverse=False)
            return x
    return sample_fn, density_fn


def compute_loss(args, model, growth_model):
    """
    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth / 
    death rate defined as a variation from uniform.
    """

    # Backward pass accumulating losses, previous state and deltas
    deltas = []
    xs = []
    prev_xs = []
    for i, (itp, tp) in enumerate(zip(int_tps[::-1], timepoints[::-1])): # tp counts down from last
        integration_times = torch.tensor([itp-args.time_length, itp]).type(torch.float32).to(device)

        # load data
        x = train_sampler(tp)
        x = torch.from_numpy(x).type(torch.float32).to(device)
        xs.append(x)
        if i > 0:
            x = torch.cat((z, x))
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to previous timepoint
        z, delta_logp = model(x, zero, integration_times=integration_times)
        prev_xs.append(z[-args.batch_size:])
        deltas.append(delta_logp)

    # compute log growth probability
    xs = torch.cat(xs)
    #growth_zs, growth_delta_logps = growth_model(xs, torch.zeros(xs.shape[0], 1).to(xs)) # Use default timestep
    #growth_logpzs = uniform_logprob(growth_zs).sum(1, keepdim=True)
    #growth_logpzs = standard_normal_logprob(growth_zs).sum(1, keepdim=True)
    #growth_logpxs = growth_logpzs - growth_delta_logps

    # compute log q(z) with forward pass
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)
    logps = [logpz]
    
    # build growth rates
    log_growthrates = [torch.zeros_like(logpz)]
    for x_state in prev_xs[::-1]:
        log_growthrates.append(growth_model(x_state))

    losses = []
    for delta_logp in deltas[::-1]:
        logpx = logps[-1] - delta_logp
        logps.append(logpx[:-args.batch_size])
        losses.append(-torch.mean(logpx[-args.batch_size:] + torch.log(log_growthrates[-1])))
    # weights = torch.tensor([1,1,1,1,1]).to(logpx)
    #weights = torch.tensor([2,1]).to(logpx)
    weights = torch.tensor([5,4,3,2,1]).to(logpx)
    losses = torch.stack(losses)

    losses = torch.mean(losses * weights)

    # Direction regularization
    if args.vecint:
        similarity_loss = 0
        for i, (itp, tp) in enumerate(zip(int_tps, timepoints)):
            itp = torch.tensor(itp).type(torch.float32).to(device)
            x = dir_train_sampler(tp)
            x = torch.from_numpy(x).type(torch.float32).to(device)
            y,z = torch.split(x, 2, dim=1)
            y = y + torch.randn_like(y) * 0.1
            # This is really hacky but I don't know a better way (alex)
            direction = model.chain[0].odefunc.odefunc.diffeq(itp, y)
            similarity_loss += 1 - torch.mean(F.cosine_similarity(direction, z))
        print(similarity_loss)
        losses += similarity_loss * args.vecint

    #loss = loss + vec_reg_loss


    #growth_losses = -torch.mean(growth_logpxs)
    #alpha = torch.tensor(args.alpha).to(growth_losses)
    #loss = (1 - alpha) * losses + alpha * growth_losses
    #loss = losses + growth_losses
    return losses#, growth_losses
    #return loss

def train(args, model, growth_model):
    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = optim.Adam(set(model.parameters()) | set(growth_model.parameters()), 
                           lr=args.lr, weight_decay=args.weight_decay)
    #growth_optimizer = optim.Adam(growth_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')
    model.train()
    growth_model.train()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        #growth_optimizer.zero_grad()

        ### Train
        if args.spectral_norm: spectral_norm_power_iteration(model, 1)
        #if args.spectral_norm: spectral_norm_power_iteration(growth_model, 1)

        #loss = compute_loss(args, model, growth_model)
            

        loss = compute_loss(args, model, growth_model)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            # Only regularize on the last timepoint
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
            )
            loss = loss + reg_loss

        #if len(growth_regularization_coeffs) > 0:
        #    growth_reg_states = get_regularization(growth_model, growth_regularization_coeffs)
        #    reg_loss = sum(
        #        reg_state * coeff for reg_state, coeff in zip(growth_reg_states, growth_regularization_coeffs) if coeff != 0
        #    )
        #    loss2 = loss2 + reg_loss

        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)

        loss.backward()
        #loss2.backward()
        optimizer.step()
        #growth_optimizer.step()

        ### Eval
        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            )
        )
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)

        logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                growth_model.eval()
                test_loss = compute_loss(args, model, growth_model)
                test_nfe = count_nfe(model)
                log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, test_loss, test_nfe)
                logger.info(log_message)

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(args.save, 'checkpt.pth'))
                model.train()

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                model.eval()
                for i, _ in enumerate(timepoints):
                    p_samples = viz_sampler(i)
                    sample_fn, density_fn = get_transforms(model, int_tps[:i+1])
                    #growth_sample_fn, growth_density_fn = get_transforms(growth_model, int_tps[:i+1])
                    plt.figure(figsize=(9, 3))
                    visualize_transform(
                        p_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
                        samples=True, npts=800, device=device
                    )
                    fig_filename = os.path.join(args.save, 'figs', '{:04d}_{:01d}.jpg'.format(itr, i))
                    utils.makedirs(os.path.dirname(fig_filename))
                    plt.savefig(fig_filename)
                    plt.close()

                    # Visualize growth transform
                    #visualize_transform(
                    #    p_samples, torch.rand, uniform_logprob, transform=growth_sample_fn, 
                    #    inverse_transform=growth_density_fn,
                    #    samples=True, npts=800, device=device
                    #)

                    #fig_filename = os.path.join(args.save, 'growth_figs', '{:04d}_{:01d}.jpg'.format(itr, i))
                    #utils.makedirs(os.path.dirname(fig_filename))
                    #plt.savefig(fig_filename)
                    #plt.close()

                model.train()
        end = time.time()
    logger.info('Training has finished.')


def plot_output(args, model):
    save_traj_dir = os.path.join(args.save, 'trajectory')
    logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    data_samples = full_sampler()
    #save_vectors(model, torch.tensor(inf_sampler(data[labels==0], batch_size=100)).type(torch.float32), args.save, device=device, end_times=int_tps, ntimes=100)
    #save_trajectory(model, data_samples, save_traj_dir, device=device, end_times=int_tps, ntimes=25)
    #trajectory_to_video(save_traj_dir)

    density_dir = os.path.join(args.save, 'density2')
    save_trajectory_density(model, data_samples, density_dir, device=device, end_times=int_tps, ntimes=100, memory=0.1)
    trajectory_to_video(density_dir)

    #save_traj_dir2 = os.path.join(args.save, 'trajectory_to_end')
    #save_trajectory(model, data_samples, save_traj_dir2, device=device, end_times=[int_tps[-1]], ntimes=25)
    #trajectory_to_video(save_traj_dir2)

class GrowthNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) + 1
        return x

if __name__ == '__main__':
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    growth_regularization_fns, growth_regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, 2, regularization_fns).to(device)
    growth_model = GrowthNet().to(device)
    #growth_model = build_model_tabular(args, 2, growth_regularization_fns).to(device)
    if args.spectral_norm: add_spectral_norm(model)
    #if args.spectral_norm: add_spectral_norm(growth_model)
    set_cnf_options(args, model)
    #set_cnf_options(args, growth_model)

    if args.test:
        model.load_state_dict(torch.load(args.save + '/checkpt.pth')['state_dict'])
        #growth_model.load_state_dict(torch.load(args.save + '/checkpt.pth')['growth_state_dict'])
    else:
        train(args, model, growth_model)

    plot_output(args, model)
    #plot_output(args, model, growth_model)


