import numpy as np
import numpy.random as random
import os
import torch
import scipy
from torch import nn
import torchvision
import clip  # installation please follow: https://github.com/openai/CLIP
import pyamg
from pyamg import coarse_grid_solver
from pyamg.aggregation import smoothed_aggregation_solver
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from typing import Dict, Tuple
import torch.nn.functional as F
import kornia as K

# Random seed
seed = 12345
seed = int(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

# Main parameters
nelx = 256
nely = 256
volfrac = 0.15
epch = 1000

sp_rate = 3
pl_rate = 4
penal = 2.5
do_fea = True

# Misc parameters
Emin = 0  # 1e-6
Emax = 1.0
cgs = coarse_grid_solver('cg')
bs = nelx * nely
ndof = 2 * (nelx + 1) * (nely + 1)

# Load
device = torch.device('cuda')
img = torch.randn(3, nely, nelx)

# Plot
plt.ion()
fig = plt.figure(figsize=(16, 4))
ax1 = fig.add_subplot(141)
ax1.axis('off')
im1 = ax1.imshow(np.zeros((nely * sp_rate, nelx * sp_rate)), cmap='gray_r', norm=colors.Normalize(vmin=0, vmax=1))

ax2 = fig.add_subplot(142)
ax2.axis('off')
im2 = ax2.imshow(np.zeros((nely * sp_rate, nelx * sp_rate, 3)))

viridis = matplotlib.colormaps['Blues']
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
newcolors[100:150, :] = pink
newcmp = ListedColormap(newcolors)

ax3 = fig.add_subplot(143)
ax3.axis('off')
im3 = ax3.imshow(np.zeros((nely, nelx)), cmap=newcmp, norm=colors.Normalize(vmin=0, vmax=1))

ax4 = fig.add_subplot(144)
ax4.axis('off')
im4 = ax4.imshow(np.zeros((nely // pl_rate, nelx // pl_rate)), cmap='jet', interpolation='none',
                 norm=colors.Normalize(vmin=0, vmax=3),
                 alpha=np.zeros((nely // pl_rate, nelx // pl_rate)))
plt.tight_layout()


# Optimization space
def inputfield(nelx, nely, sp_rate):
    nelx = nelx * sp_rate
    nely = nely * sp_rate
    edofMat = np.zeros((nelx * nely, 8), dtype=int)

    for elx in range(nelx):
        for ely in range(nely):
            el = elx + ely * nelx
            n1 = elx + ely * (nelx + 1)
            n2 = elx + (ely + 1) * (nelx + 1)
            edofMat[el, :] = np.array(  # DOF ids of each element
                [2 * n1, 2 * n1 + 1, 2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1])

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    xy_norm = np.max([nelx, nely])
    xv, yv = np.meshgrid(np.linspace(0, nelx, nelx) / xy_norm, np.linspace(0, nely, nely) / xy_norm)
    points = np.zeros((nelx * nely, 2))
    points[:, 0] = xv.reshape(nelx * nely)
    points[:, 1] = yv.reshape(nelx * nely)
    t_points = torch.from_numpy(points).float().to(device)
    return t_points, edofMat, iK, jK


# Element stiffness matrix
def lk():
    E = 1
    nu = 0.3
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

    B = (1 / 2) * np.array([[-1, 0, 1, 0, 1, 0, -1, 0],
                            [0, -1, 0, -1, 0, 1, 0, 1],
                            [-1, -1, -1, 1, 1, 1, 1, -1]])
    C = (E / (1 - nu ** 2)) * np.array([[1, nu, 0],
                                        [nu, 1, 0],
                                        [0, 0, (1 - nu) / 2]])

    return KE, B, C


# Boundary condition
def bc(nelx, nely, task, sp_rate):
    nelx = nelx * sp_rate
    nely = nely * sp_rate
    ndof = 2 * (nelx + 1) * (nely + 1)
    dofs = np.arange(ndof)

    f = np.zeros((ndof, 1))
    # MBB
    if task == 'MBB':
        fixed = np.union1d(dofs[0:2 * nely * (nelx + 1):2 * (nelx + 1)], np.array([ndof - 1]))  # 0->X, 1->Y
        f[1, 0] = -1
    # Bridge
    elif task == 'Bridge':
        fixed = np.union1d(dofs[0:2 * nely * (nelx + 1):2 * (nelx + 1)], np.array([ndof - 1]))
        f[2 * nely * (nelx + 1) - 1:2 * (nelx + 1) * (nely + 1):2, 0] = -5e-2
    # L-Bracket
    elif task == 'L-Bracket':
        fixed = dofs[0:2 * (nelx + 1):1]
        f[ndof - 1 - int(0.36 * nely) * 2 * (nelx + 1)] = 1

    free = np.setdiff1d(dofs, fixed)
    return f, free


# Finite element analysis
def FEA(nelx, nely, KE, x, penal, iK, jK, edofMat, f, free, epoch, U_last):
    bs = nelx * nely

    sK = ((KE.flatten()[np.newaxis]).T @ (Emin + (x.cpu().detach().numpy()) ** penal * (Emax - Emin)).T).flatten(
        order='F')
    K = scipy.sparse.csr_matrix((sK, (iK, jK)), shape=(ndof, ndof))
    K_free = K[free, :][:, free]
    U = np.zeros((ndof, 1))

    if epoch == 0:
        U[free, 0] = pyamg.krylov.cg(K_free, f[free, 0], tol=1e-6)[0]
        U_last = U[free, 0]
    else:
        ml = smoothed_aggregation_solver(K_free, coarse_solver=cgs)
        M = ml.aspreconditioner(cycle='V')
        U[free, 0], info = pyamg.krylov.cg(K_free, f[free, 0], tol=1e-6, M=M, x0=U_last)
        U_last = U[free, 0]

    # Solve ce_tensor, ce_tensor is the element compliance
    ce = (np.dot(U[edofMat].reshape((nelx * nely, 8)), KE) * U[edofMat].reshape((nelx * nely, 8))).sum(1)
    dc = (((-penal * x.cpu().detach().numpy() ** (penal - 1) * (Emax - Emin)).flatten()) * ce).reshape(bs, 1)

    # Construct object function. Penalty function is very unsatable and often fall into local minimum. Checked'
    compliance = (((Emin + x.cpu().detach().numpy() ** penal * (Emax - Emin)).flatten()) * ce).sum()
    return U, U_last, dc, compliance


# Stress computation. Just a minor test for visualization, not mentioned in the paper, correctness not guaranteed.
def stress_cal(x, nelx, nely, B, C, U, edofMat):
    U = U.reshape(U.shape[0], 1)

    edMat = np.zeros((nelx * nely, 8), dtype=int)
    for ely in range(nely):
        for elx in range(nelx):
            el = ely + elx * nely
            n1 = ely + (elx + 0) * (nely + 1)
            n2 = ely + (elx + 1) * (nely + 1)
            edMat[el, :] = np.array(
                [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3])

    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = ely + elx * (nely + 1)
            n2 = ely + (elx + 1) * (nely + 1)
            edofMat[el, :] = np.array(  # DOF ids of each element
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2 + 0, 2 * n2 + 1, 2 * n1 + 0, 2 * n1 + 1])

    Ue = U[edMat]
    stress = (C @ B @ Ue).reshape(nelx * nely, 3)
    stress_final = np.sqrt(
        stress[:, 0].reshape(-1) ** 2 + stress[:, 1].reshape(-1) ** 2 + 3 * stress[:, 2].reshape(-1) ** 2 - stress[:,
                                                                                                            0].reshape(
            -1) * stress[:, 1].reshape(-1))
    stress_final = stress_final  # * x.reshape(-1)
    return stress_final


# Stress computation. Just a minor test for visualization, not mentioned in the paper, correctness not guaranteed.
def stress_cal_t(x, nelx, nely, B, C, U, edofMat):
    if not isinstance(U, torch.Tensor):
        U = torch.tensor(U, dtype=torch.float32, requires_grad=True).to(device)

    U = U.view(U.shape[0], 1)

    edMat = torch.zeros((nelx * nely, 8), dtype=torch.int).to(device)
    for ely in range(nely):
        for elx in range(nelx):
            el = ely + elx * nely
            n1 = ely + (elx + 0) * (nely + 1)
            n2 = ely + (elx + 1) * (nely + 1)
            edMat[el, :] = torch.tensor(
                [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3],
                dtype=torch.int)

    edofMat = torch.zeros((nelx * nely, 8), dtype=torch.int).to(device)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = ely + elx * (nely + 1)
            n2 = ely + (elx + 1) * (nely + 1)
            edofMat[el, :] = torch.tensor(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2 + 0, 2 * n2 + 1, 2 * n1 + 0, 2 * n1 + 1],
                dtype=torch.int).to(device)

    Ue = U[edMat]
    stress = torch.matmul(
        torch.matmul(torch.tensor(C, dtype=torch.float32).to(device), torch.tensor(B, dtype=torch.float32).to(device)),
        Ue)
    stress = stress.view(nelx * nely, 3)
    stress_final = torch.sqrt(stress[:, 0] ** 2 + stress[:, 1] ** 2 + 3 * stress[:, 2] ** 2 -
                              stress[:, 0] * stress[:, 1])
    stress_final = stress_final  # * x.view(-1)
    return stress_final


# Hash-encoding
class HashEncode2D(nn.Module):
    def __init__(self, l=16, t=2 ** 19, f=2, n_min=8, n_max=128):  # nmin, nmax: [8, 256]
        super().__init__()
        self.l = l
        self.t = t
        self.f = f

        # 1. Random initialize a hash table
        self.register_buffer('primes', torch.tensor([1, 2654435761]))
        self.hash_table = nn.Parameter(torch.rand([l, t, f], requires_grad=True) * 2e-4 - 1e-4)

        # 2. Multi-res grids
        b = np.exp((np.log(n_max) - np.log(n_min)) / (l - 1))
        self.ns = [int(n_min * (b ** i)) for i in range(l)]

    @property
    def encoded_vector_size(self):
        return self.l * self.f

    def get_features(self, coord, layer, n):  # (n - 1)  (n + 1)
        # 3. Grid index searching
        index1 = (coord // (1 / n)).to(dtype=torch.long)
        index2 = (coord // (1 / n) + torch.Tensor([1, 0]).reshape(1, 2, 1, 1).to(device)).to(dtype=torch.long)
        index3 = (coord // (1 / n) + torch.Tensor([0, 1]).reshape(1, 2, 1, 1).to(device)).to(dtype=torch.long)
        index4 = (coord // (1 / n) + torch.Tensor([1, 1]).reshape(1, 2, 1, 1).to(device)).to(dtype=torch.long)

        coord_vert1_org = index1 * (1 / n)
        coord_vert2_org = index2 * (1 / n)
        coord_vert3_org = index3 * (1 / n)
        coord_vert4_org = index4 * (1 / n)

        coord_vert1 = (coord * n).to(dtype=torch.long)
        coord_vert2 = (coord * n + 1 * torch.Tensor([1, 0]).reshape(1, 2, 1, 1).to(device)).to(dtype=torch.long)
        coord_vert3 = (coord * n + 1 * torch.Tensor([0, 1]).reshape(1, 2, 1, 1).to(device)).to(dtype=torch.long)
        coord_vert4 = (coord * n + 1 * torch.Tensor([1, 1]).reshape(1, 2, 1, 1).to(device)).to(dtype=torch.long)

        # 4. Hashing
        id1 = coord_vert1 * self.primes.view([2, 1, 1])
        id1 = (id1[:, 0] ^ id1[:, 1]) % self.t

        id2 = coord_vert2 * self.primes.view([2, 1, 1])
        id2 = (id2[:, 0] ^ id2[:, 1]) % self.t

        id3 = coord_vert3 * self.primes.view([2, 1, 1])
        id3 = (id3[:, 0] ^ id3[:, 1]) % self.t

        id4 = coord_vert4 * self.primes.view([2, 1, 1])
        id4 = (id4[:, 0] ^ id4[:, 1]) % self.t

        f1 = self.hash_table[layer, id1].permute(0, 3, 1, 2)
        f2 = self.hash_table[layer, id2].permute(0, 3, 1, 2)
        f3 = self.hash_table[layer, id3].permute(0, 3, 1, 2)
        f4 = self.hash_table[layer, id4].permute(0, 3, 1, 2)

        # 4. Bilinear interpolate
        r1 = (f1 * (coord_vert2_org[:, 0, :, :] - coord[:, 0, :, :]) + f2 * (
                coord[:, 0, :, :] - coord_vert1_org[:, 0, :, :])) * n
        r2 = (f3 * (coord_vert4_org[:, 0, :, :] - coord[:, 0, :, :]) + f4 * (
                coord[:, 0, :, :] - coord_vert3_org[:, 0, :, :])) * n
        rf = (r1 * (coord_vert3_org[:, 1, :, :] - coord[:, 1, :, :]) + r2 * (
                coord[:, 1, :, :] - coord_vert1_org[:, 1, :, :])) * n
        return rf

    def forward(self, coord):
        features = torch.hstack([self.get_features(coord, layer, n) for layer, n in enumerate(self.ns)])
        return features


# MLP
class Model(nn.Module):
    def __init__(self, encoder, num_planes=64, num_layers=2):
        super().__init__()
        self.enc = encoder

        # 1x1 convolution is equivalent to MLP for a point in the 2D-coordinates
        layers = [nn.Conv2d(encoder.encoded_vector_size, num_planes, 1)]
        for _ in range(num_layers - 2):
            layers += [nn.ReLU(),
                       nn.Conv2d(num_planes, num_planes, 1)]
        layers += [nn.ReLU(),
                   nn.Conv2d(num_planes, 4, 1),
                   nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.enc(x)
        out = self.mlp(feature)
        return out


# Training process
def train(name, img, model, text, task, task_idx, steps=201, output_visualize=True):
    _, h, w = img.size()
    optimizer = torch.optim.Adam([
        {'params': model.enc.parameters()},
        {'params': model.mlp.parameters(), 'weight_decay': 1e-6}
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)
    model.to(device)

    # TO
    KE, B, C = lk()
    f, free = bc(nelx, nely, task, sp_rate=1)
    t_points, edofMat, iK, jK = inputfield(nelx, nely, sp_rate=1)
    t_points = torch.permute(t_points.reshape(nely, nelx, 2, 1), (3, 2, 0, 1))
    t_points_h, edofMat_h, iK_h, jK_h = inputfield(nelx, nely, sp_rate)
    t_points_h = torch.permute(t_points_h.reshape(nely * sp_rate, nelx * sp_rate, 2, 1), (3, 2, 0, 1))

    # Augmentation operator
    randgray = torchvision.transforms.RandomGrayscale(0.1)  # 0.1(default) 0.4 0.7 0.1
    randcrop = torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3), antialias=False)
    randaffine = torchvision.transforms.RandomAffine(45, translate=(0, 0.1), scale=(0.9, 1.1))
    gaussianblur = torchvision.transforms.GaussianBlur(kernel_size=25, sigma=1)

    # Collect data
    comps = []
    vols = []
    semloss = []
    for i in range(steps):
        optimizer.zero_grad()

        # Forward
        pred = model(t_points)
        pred = torch.permute(pred, (2, 3, 1, 0)).reshape(nely * nelx, 4)
        x = pred[:, 0].reshape(nelx * nely, 1)

        # L-bracket
        if task == 'L-Bracket':
            x = x.reshape((nely, nelx))
            x.data[0:int(0.6 * nely), int(0.4 * nelx):] = 0
            x = x.reshape(nelx * nely, 1)

        c = pred[:, 1:4]
        alpha = x.repeat(1, 3)
        c = c * alpha ** 1  # + (1 - alpha)  # (1 - alpha) is to ensure white background

        # Connected-component labelling (CCL)
        with torch.no_grad():
            x_ccl = torch.clone(pred[:, 0]).reshape(1, 1, nely, nelx)
            x_ccl[x_ccl > 0.3] = 1  # 0.1
            x_ccl[x_ccl <= 0.3] = 0
            x_labels = K.contrib.connected_components(x_ccl, num_iterations=1000) / (nelx * nely)

            mask_ids = torch.unique(x_labels)
            mask_idx = []
            for nums in range(torch.unique(x_labels).shape[0]):
                mask_idx.append((x_labels == mask_ids[nums]).sum())

            id_bg = x_labels == mask_ids[0]
            id_to = x_labels == mask_ids[-1]

            x_labels[x_labels == mask_ids[0]] = 0
            x_labels[x_labels == mask_ids[-1]] = 1
            x_labels[~(id_bg + id_to)] = 0

        x_ccl_loss = pred[:, 0].reshape(1, 1, nely, nelx)
        ccl_loss = 1.0 * ((x_ccl_loss[~(id_bg + id_to)]) ** 1).sum()  # 1 is good
        ccl_loss.backward(retain_graph=True)

        # Random background
        rd_bg = torch.rand(1, 3, nelx, nely).to(device)
        rd_bg = gaussianblur(rd_bg)
        rd_bg = rd_bg.reshape(nelx * nely, 3)  # consider adding gauss-noise to random bg
        c = c + rd_bg * (1 - alpha ** 1)

        # Augmentation
        img = torchvision.transforms.functional.resize(img=c.view(1, nelx, nely, 3).permute(0, 3, 1, 2),
                                                       size=(224, 224), antialias=False)
        naug = 16
        imgs_aug = []
        for j in range(naug):
            imgs_aug.append(randaffine(randcrop(randgray(img))))
        imgs_aug = torch.cat(imgs_aug, 0)
        img_feature = model_clip.encode_image(imgs_aug)
        text_feature = model_clip.encode_text(clip.tokenize(text).to(device))  # 0.02GB
        clip_loss = -9e3 * torch.cosine_similarity(text_feature, img_feature, dim=-1).mean()
        clip_loss.backward(retain_graph=True)

        # Pooling
        if i == 0:
            temp0, edofMat, iK, jK = inputfield(nelx // pl_rate, nely // pl_rate, sp_rate=1)
            f, free = bc(nelx // pl_rate, nely // pl_rate, task, sp_rate=1)
            U_last = np.zeros((2 * (nelx // pl_rate + 1) * (nely // pl_rate + 1), 1))
        rho = nn.functional.avg_pool2d(x.reshape(1, 1, nelx, nely), pl_rate).reshape(-1, 1)

        # FEA
        U, U_last, dc, compliance = FEA(nelx // pl_rate, nely // pl_rate, KE, rho, penal, iK, jK, edofMat, f, free, i,
                                        U_last)
        volume_loss = 3e3 * (torch.mean(rho) - volfrac) ** 2
        dv = torch.autograd.grad(volume_loss, rho)[0]
        fea_loss = (rho * (dv + torch.Tensor(dc).to(device))).sum()
        fea_loss.backward()

        # Stress
        stress = stress_cal(rho.cpu().detach().numpy(), nelx // pl_rate, nely // pl_rate, B, C, U, edofMat)

        # Optimize
        optimizer.step()
        comps.append(compliance)
        vols.append(x.mean().cpu().detach().numpy())
        semloss.append((clip_loss / (-6e3)).cpu().detach().numpy())
        print(i, compliance, x.mean().cpu().detach().numpy())

        with torch.no_grad():
            pred_h = model(t_points_h)
            pred_h = torch.permute(pred_h, (2, 3, 1, 0)).reshape(nely * nelx * sp_rate ** 2, 4)
            x_h = pred_h[:, 0].reshape(nelx * nely * sp_rate ** 2, 1)

            # L-bracket
            if task == 'L-Bracket':
                x_h = x_h.reshape((nelx * sp_rate, nely * sp_rate))
                x_h.data[0:int(0.6 * nely) * sp_rate, int(0.4 * nelx * sp_rate):] = 0
                x_h = x_h.reshape(nelx * nely * sp_rate ** 2, 1)

            c_h = pred_h[:, 1:4]
            alpha_h = x_h.repeat(1, 3)
            c_h = c_h * alpha_h + (1 - alpha_h)

        if output_visualize:
            img_x = x_h.cpu().detach().numpy()
            img_x = img_x.reshape(nely * sp_rate, nelx * sp_rate)
            im1.set_array(img_x)

            img_c = c_h.cpu().detach().numpy()
            img_c = img_c.reshape(nely * sp_rate, nelx * sp_rate, 3)
            im2.set_array(img_c)

            x_labels[~(id_bg + id_to)] = 0.5
            img_l = x_labels.cpu().detach().numpy()
            img_l = img_l.reshape(nely, nelx)
            im3.set_array(img_l)

            im4.set_array(stress.cpu().detach().numpy().reshape((nely // pl_rate, nelx // pl_rate)))
            im4.set_alpha(rho.cpu().detach().numpy().reshape((nely // pl_rate, nelx // pl_rate)))

            plt.pause(0.01)

        # if i % 100 == 0:
        #     plt.savefig('../../data/Tests/2D' + str(task_idx) + '_' + str(i) + '.png')
        #     torch.save(model.state_dict(), '../../data/Tests/2D' + str(task_idx) + '.pkl')
        #
        #     np.save('../../data/Tests/2D' + str(task_idx) + '_comp.npy', comps)
        #     np.save('../../data/Tests/2D' + str(task_idx) + '_vols.npy', vols)
        #     np.save('../../data/Tests/2D' + str(task_idx) + '_seml.npy', semloss)
        #
        #     extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #     fig.savefig('../../data/Images/2D' + str(task_idx) + '_ax1.png', bbox_inches=extent, transparent=True)
        #     extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #     fig.savefig('../../data/Images/2D' + str(task_idx) + '_ax2.png', bbox_inches=extent, transparent=True)
        #     extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #     fig.savefig('../../data/Images/2D' + str(task_idx) + '_ax3.png', bbox_inches=extent, transparent=True)
        #     extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #     fig.savefig('../../data/Images/2D' + str(task_idx) + '_ax4.png', bbox_inches=extent, transparent=True)


# ================================================================================
# Try running the program
if __name__ == "__main__":
    model = Model(encoder=HashEncode2D())
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    train('hash', img, model, "golden, Baroque style", 'Bridge', 1)
