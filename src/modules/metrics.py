from copy import deepcopy

import seaborn as sns

import torch
from pyhessian import hessian
import matplotlib.pyplot as plt


def acc_metric(y_pred, y_true):
    correct = (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
    acc = correct / y_pred.size(0)
    return acc


def prepare_evaluators(y_pred, y_true, loss):
    acc = acc_metric(y_pred, y_true)
    evaluators = {'loss': loss.item(), 'acc': acc}
    return evaluators

def entropy_loss(y_pred):
    return -torch.sum(torch.nn.functional.softmax(y_pred, dim=1) * torch.log_softmax(y_pred, dim=1))


class BatchVariance(torch.nn.Module):
    def __init__(self, model, optim, criterion=None, dataloader=None, device=None):
        super().__init__()
        self.model_zero = deepcopy(model)
        self.model = model
        self.optim = optim
        self.criterion = criterion
        # held out % examples from dataloader
        self.dataloader = dataloader
        self.device = device
        self.model_trajectory_length = 0.0

    def forward(self, evaluators, distance_type):
        lr = self.optim.param_groups[-1]['lr']
        norm = self.model_gradient_norm()
        evaluators['model_gradient_norm_squared'] = norm ** 2
        self.model_trajectory_length += lr * norm
        evaluators['model_trajectory_length'] = self.model_trajectory_length
        distance_from_initialization = self.distance_between_models(distance_type)
        evaluators[f'distance_from_initialization_{distance_type}'] = distance_from_initialization
        evaluators['excessive_length'] = evaluators['model_trajectory_length'] - evaluators[f'distance_from_initialization_{distance_type}']
        return evaluators
        

    def model_gradient_norm(self, norm_type=2.0):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        norm = torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in parameters]), norm_type)
        return norm.item()
    
    def distance_between_models(self, distance_type):
        def distance_between_models_l2(parameters1, parameters2, norm_type=2.0):
            """
            Returns the l2 distance between two models.
            """
            distance = torch.norm(torch.stack([torch.norm(p1-p2, norm_type) for p1, p2 in zip(parameters1, parameters2)]), norm_type)
            return distance.item()
        
        def distance_between_models_cosine(parameters1, parameters2):
            """
            Returns the cosine distance between two models.
            """
            distance = 0
            for p1, p2 in zip(parameters1, parameters2):
                distance += 1 - torch.cosine_similarity(p1.flatten(), p2.flatten())
            return distance.item()

        """
        Returns the distance between two models.
        """
        parameters1 = [p for p in self.model_zero.parameters() if p.requires_grad]
        parameters2 = [p for p in self.model.parameters() if p.requires_grad]
        if distance_type == 'l2':
            distance = distance_between_models_l2(parameters1, parameters2)
        elif distance_type == 'cosine':
            distance = distance_between_models_cosine(parameters1, parameters2)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        return distance
    
    def sharpness(self, dataloader, maxIter=100):
        hessian_comp = hessian(self.model, self.criterion, dataloader=dataloader, cuda=self.device.type!='cpu')
        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=maxIter)
        self.model.train()
        return top_eigenvalues[0].item()


class CosineAlignments:
    def __init__(self, model, loader, criterion) -> None:
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.device = next(model.parameters()).device

    def calc_variance(self, n):
        gs = torch.tensor(self.gather_gradients(n))
        gdv = 0.
        for i in range(n):
            for j in range(i+1, n):
                gdv += 1 - torch.dot(gs[i], gs[j]) / torch.norm(gs[i], gs[j])
        gdv /= 2 / (n * (n - 1))
        return gdv


    def gather_gradients(self, n, device):
        gs = []
        for i, (x_true, y_true) in enumerate(self.loader):
            if i >= n: break
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            self.criterion(y_pred, y_true).backward()
            g = [p.grad for p in self.model.parameters() if p.requires_grad]
            gs.append(g)
            self.model.zero_grad()
        return gs


import torch
from torch.func import functional_call, vmap, grad

# class PerSampleGrad(torch.nn.Module):
#     # compute loss and grad per sample 
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
#         self.ft_criterion = vmap(grad(self.compute_loss), in_dims=(None, None, 0, 0))

#     def forward(self, x_true, y_true):
#         similarity_matrices = {}
#         graham_matrices = {}
#         cov_matrices = {}
#         trace_of_cov_matrices = {}
#         rank_of_gradients = {}
#         params = {k: v.detach() for k, v in self.model.named_parameters()}
#         buffers = {k: v.detach() for k, v in self.model.named_buffers()}
#         ft_per_sample_grads = self.ft_criterion(params, buffers, x_true, y_true)
#         # ft_per_sample_grads2 = self.ft_criterion(params, buffers, x_true2, y_true2)
#         ft_per_sample_grads = {k: v.detach().data for k, v in ft_per_sample_grads.items()}
#         # ft_per_sample_grads2 = {k: v.detach().data for k, v in ft_per_sample_grads2.items()}
#         concatenated_weights = torch.empty((x_true.shape[0], 0), device=x_true.device)
#         for k in ft_per_sample_grads:
#             # rank_of_gradients[k] = torch.matrix_rank(ft_per_sample_grads[k])
#             ft_per_sample_grads[k] = ft_per_sample_grads[k].reshape(x_true.shape[0], -1)
#             # trace_of_cov_matrices[k] = self.trace_of_cov(ft_per_sample_grads[k])
#             concatenated_weights = torch.cat((concatenated_weights, ft_per_sample_grads[k]), dim=1)
#             normed_ft_per_sample_grad = ft_per_sample_grads[k] / (1e-9 + torch.norm(ft_per_sample_grads[k], dim=1, keepdim=True))
#             similarity_matrices[k] = normed_ft_per_sample_grad @ normed_ft_per_sample_grad.T
#             graham_matrices[k] = ft_per_sample_grads[k] @ ft_per_sample_grads[k].T / ft_per_sample_grads[k].shape[0]
#             cov_matrices[k] = (ft_per_sample_grads[k] - ft_per_sample_grads[k].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads[k] - ft_per_sample_grads[k].mean(dim=0, keepdim=True)).T / ft_per_sample_grads[k].shape[0]
#         ft_per_sample_grads['concatenated_weights'] = concatenated_weights
#         trace_of_cov_matrices['concatenated_weights'] = self.trace_of_cov(ft_per_sample_grads['concatenated_weights'])
#         normed_concatenated_weights = ft_per_sample_grads['concatenated_weights'] / (1e-9 + torch.norm(ft_per_sample_grads['concatenated_weights'], dim=1, keepdim=True))
#         similarity_matrices['concatenated_weights'] = normed_concatenated_weights @ normed_concatenated_weights.T
#         graham_matrices['concatenated_weights'] = ft_per_sample_grads['concatenated_weights'] @ ft_per_sample_grads['concatenated_weights'].T
#         cov_matrices['concatenated_weights'] = (ft_per_sample_grads['concatenated_weights'] - ft_per_sample_grads['concatenated_weights'].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads['concatenated_weights'] - ft_per_sample_grads['concatenated_weights'].mean(dim=0, keepdim=True)).T
#         return similarity_matrices, graham_matrices, cov_matrices, trace_of_cov_matrices
    
#     def compute_loss(self, params, buffers, sample, target):
#         batch = sample.unsqueeze(0)
#         targets = target.unsqueeze(0)
#         predictions = functional_call(self.model, (params, buffers), (batch,))
#         loss = self.criterion(predictions, targets)
#         return loss
    
#     def trace_of_cov(self, g):
#         g_mean = torch.mean(g, dim=0)
#         g = g - g_mean
#         tr = torch.mean(g.norm(dim=1)**2)
#         return tr.item()
    
            
# # wyliczyć sharpness dla macierzy podobieństwa, loader składa się z 500 przykładów
    
# class Stiffness(torch.nn.Module):
#     # add option to compute loss directly
#     # add option with train-val
#     def __init__(self, model, num_classes, x_data, y_data, logger=None, every_nb=5):
#         super().__init__()
#         self.per_sample_grad = PerSampleGrad(model)
#         self.num_classes = num_classes
#         idxs = torch.arange(0, x_data.shape[0], every_nb)
#         self.x_data = x_data[idxs]
#         self.y_data = y_data[idxs]
#         self.logger = logger
        
#     def run_stiffness(self, step):
#         stifness_heatmaps = {}
#         stiffness_logs = {}
#         similarity_matrices, graham_matrices, cov_matrices, trace_of_cov_matrices, expected_stiffness_cosine, expected_stiffness_sign, c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine, c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign = self.forward(self.x_data, self.y_data)
        
#         stiffness_logs['stiffness_sharpness/similarity'] = self.sharpness(similarity_matrices['concatenated_weights'])
#         stiffness_logs['stiffness_sharpness/graham'] = self.sharpness(graham_matrices['concatenated_weights'])
#         stiffness_logs['stiffness_sharpness/cov'] = self.sharpness(cov_matrices['concatenated_weights'])
#         stiffness_logs['stiffness_sharpness/trace_of_cov'] = trace_of_cov_matrices['concatenated_weights']
        
#         fig1, ax1  = plt.subplots(1, 1, figsize=(10, 10))
#         fig2, ax2  = plt.subplots(1, 1, figsize=(10, 10))
#         fig3, ax3  = plt.subplots(1, 1, figsize=(10, 10))
#         stifness_heatmaps['stiffness/alignment_all'] = sns.heatmap(similarity_matrices['concatenated_weights'].data.cpu().numpy(), ax=ax1).get_figure()
#         # stifness_heatmaps['stiffness/graham_all'] = sns.heatmap(graham_matrices['concatenated_weights'].data.cpu().numpy(), annot=True).get_figure()
#         # stifness_heatmaps['stiffness/cov_all'] = sns.heatmap(cov_matrices['concatenated_weights'].data.cpu().numpy(), annot=True).get_figure()
#         stifness_heatmaps['stiffness/class_alignment_cosine'] = sns.heatmap(c_stiffness_cosine['concatenated_weights'].data.cpu().numpy(), ax=ax2).get_figure()
#         stifness_heatmaps['stiffness/class_alignment_sign'] = sns.heatmap(c_stiffness_sign['concatenated_weights'].data.cpu().numpy(), ax=ax3).get_figure()
        
#         stiffness_logs['stiffness/overall_stiffness_cosine'] = expected_stiffness_cosine['concatenated_weights']
#         stiffness_logs['stiffness/overall_stiffness_sign'] = expected_stiffness_sign['concatenated_weights']
#         stiffness_logs['stiffness/between_classes_cosine'] = stiffness_between_classes_cosine['concatenated_weights']
#         stiffness_logs['stiffness/within_classes_cosine'] = stiffness_within_classes_cosine['concatenated_weights']
#         stiffness_logs['stiffness/between_classes_sign'] = stiffness_between_classes_sign['concatenated_weights']
#         stiffness_logs['stiffness/within_classes_sign'] = stiffness_within_classes_sign['concatenated_weights']
        
#         stiffness_logs['steps/stiffness_train'] = step
        
#         self.logger.log_figures(stifness_heatmaps, step)
#         self.logger.log_scalars(stiffness_logs, step)
        
        
#     def forward(self, x_true, y_true, x_true2=None, y_true2=None):
#         similarity_matrices, graham_matrices, cov_matrices, trace_of_cov_matrices = self.per_sample_grad(x_true, y_true) # [<g_i/|g_i|, g_j/|g_j|>]_{i,j}, [<g_i, g_j>]_{i,j}, [<g_i-g, g_j-g>]_{i,j}
#         expected_stiffness_cosine = self.cosine_stiffness(similarity_matrices) 
#         expected_stiffness_sign = self.sign_stiffness(similarity_matrices)
#         c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine  = self.class_stiffness(similarity_matrices, y_true, whether_sign=False)
#         c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign  = self.class_stiffness(similarity_matrices, y_true, whether_sign=True)
#         return similarity_matrices, graham_matrices, cov_matrices, trace_of_cov_matrices, expected_stiffness_cosine, expected_stiffness_sign, c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine, c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign
    
#     def cosine_stiffness(self, similarity_matrices):
#         expected_stiffness = {k: torch.mean(similarity_matrix).item() for k, similarity_matrix in similarity_matrices.items()}
#         return expected_stiffness
    
#     def sign_stiffness(self, similarity_matrices):
#         expected_stiffness = {k: torch.mean(torch.sign(similarity_matrix)).item() for k, similarity_matrix in similarity_matrices.items()}
#         return expected_stiffness
    
#     def class_stiffness(self, similarity_matrices, y_true, whether_sign=False):
#         c_stiffness = {}
#         # extract the indices into dictionary from y_true tensor where the class is the same
#         indices = {i: torch.where(y_true == i)[0] for i in range(self.num_classes)}
#         indices = {k: v for k, v in indices.items() if v.shape[0] > 0}
#         for k, similarity_matrix in similarity_matrices.items():
#             c_stiffness[k] = torch.zeros((self.num_classes, self.num_classes), device=y_true.device)
#             for c1, idxs1 in indices.items():
#                 for c2, idxs2 in indices.items():
#                     sub_matrix = similarity_matrix[idxs1, :][:, idxs2]
#                     sub_matrix = torch.sign(sub_matrix) if whether_sign else sub_matrix
#                     c_stiffness[k][c1, c2] = torch.mean(sub_matrix) if c1 != c2 else (torch.sum(sub_matrix) - sub_matrix.size(0)) / (sub_matrix.size(0)**2 - sub_matrix.size(0))
                    
#         stiffness_between_classes = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in c_stiffness.items()}
#         stiffness_within_classes = {k: (torch.diagonal(v).sum() / v.size(0)).item() for k, v in c_stiffness.items()}
        
#         return c_stiffness, stiffness_between_classes, stiffness_within_classes  
        
#     def sharpness(self, similarity_matrix):
#         w, _ = torch.linalg.eig(similarity_matrix)
#         max_eig = torch.max(w.real) # .abs()??
#         return max_eig
    
    
    
    
    
##############################################################################################################
    
    
# import torch
# from torch.func import functional_call, vmap, grad
    
    
# class PerSampleGrad(torch.nn.Module):
#     # compute loss and grad per sample 
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
#         self.ft_criterion = vmap(grad(self.compute_loss), in_dims=(None, None, 0, 0))

#     def forward(self, x_true1, y_true1, x_true2=None, y_true2=None):
#         matrices = {
#             'similarity': {},
#             'graham': {},
#             'cov': {},
#         }
#         scalars = { # rozróżnienie na te miary co po x_true1 i x_true2, TODO
#             'trace_of_cov': {},
#             'rank_of_gradients': {},
#             'rank_of_similarity': {},
#         }
#         params = {k: v.detach() for k, v in self.model.named_parameters()}
#         buffers = {k: v.detach() for k, v in self.model.named_buffers()}
            
#         ft_per_sample_grads1 = self.ft_criterion(params, buffers, x_true1, y_true1)
#         ft_per_sample_grads1 = {k1: v.detach().data for k1, v in ft_per_sample_grads1.items()}
#         concatenated_weights1 = torch.empty((x_true1.shape[0], 0), device=x_true1.device)
#         if x_true2 is not None:
#             ft_per_sample_grads2 = self.ft_criterion(params, buffers, x_true2, y_true2)
#             ft_per_sample_grads2 = {k2: v.detach().data for k2, v in ft_per_sample_grads2.items()}
#             concatenated_weights2 = torch.empty((x_true2.shape[0], 0), device=x_true2.device)
        
#         for k in ft_per_sample_grads1:
#             scalars['rank_of_gradients'][k] = self.matrix_rank(ft_per_sample_grads1[k]) / torch.min(ft_per_sample_grads1[k].shape[1:]) # ???
#             ft_per_sample_grads1[k] = ft_per_sample_grads1[k].reshape(x_true1.shape[0], -1)
#             normed_ft_per_sample_grad1 = ft_per_sample_grads1[k] / (1e-9 + torch.norm(ft_per_sample_grads1[k], dim=1, keepdim=True))
#             concatenated_weights1 = torch.cat((concatenated_weights1, ft_per_sample_grads1[k]), dim=1)
#             scalars['trace_of_cov'][k] = self.trace_of_cov(ft_per_sample_grads1[k])
#             if x_true2 is not None:
#                 ft_per_sample_grads2[k] = ft_per_sample_grads2[k].reshape(x_true2.shape[0], -1)
#                 concatenated_weights2 = torch.cat((concatenated_weights2, ft_per_sample_grads2[k]), dim=1)
#                 normed_ft_per_sample_grad2 = ft_per_sample_grads2[k] / (1e-9 + torch.norm(ft_per_sample_grads2[k], dim=1, keepdim=True))
#             else:
#                 ft_per_sample_grads2 = ft_per_sample_grads1
#                 normed_ft_per_sample_grad2 = normed_ft_per_sample_grad1
               
#             matrices['similarity'][k] = normed_ft_per_sample_grad1 @ normed_ft_per_sample_grad2.T
#             matrices['graham'][k] = ft_per_sample_grads1[k] @ ft_per_sample_grads2[k].T / matrices['similarity'][k].shape[0]
#             matrices['cov'][k] = (ft_per_sample_grads1[k] - ft_per_sample_grads1[k].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads2[k] - ft_per_sample_grads2[k].mean(dim=0, keepdim=True)).T / matrices['similarity'][k].shape[0]
#             scalars['rank_of_similarity'][k] = self.matrix_rank(matrices['similarity'][k], hermitian=False if x_true2 is not None else True)
            
#         scalars['rank_of_gradients']['mean'] = sum(scalars['rank_of_gradients'].values()) / len(scalars['rank_of_gradients'])
#         ft_per_sample_grads1['concatenated_weights'] = concatenated_weights1
#         normed_concatenated_weights1 = ft_per_sample_grads1['concatenated_weights'] / (1e-9 + torch.norm(ft_per_sample_grads1['concatenated_weights'], dim=1, keepdim=True))
#         scalars['trace_of_cov']['concatenated_weights'] = self.trace_of_cov(ft_per_sample_grads1['concatenated_weights'])
#         if x_true2 is not None:
#             ft_per_sample_grads2['concatenated_weights'] = concatenated_weights2
#             normed_concatenated_weights2 = ft_per_sample_grads2['concatenated_weights'] / (1e-9 + torch.norm(ft_per_sample_grads2['concatenated_weights'], dim=1, keepdim=True))
#         else:
#             normed_concatenated_weights2 = normed_concatenated_weights1
#             ft_per_sample_grads2 = ft_per_sample_grads1
        
#         matrices['similarity']['concatenated_weights'] = normed_concatenated_weights1 @ normed_concatenated_weights2.T
#         matrices['graham']['concatenated_weights'] = ft_per_sample_grads1['concatenated_weights'] @ ft_per_sample_grads2['concatenated_weights'].T
#         matrices['cov']['concatenated_weights'] = (ft_per_sample_grads1['concatenated_weights'] - ft_per_sample_grads1['concatenated_weights'].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads2['concatenated_weights'] - ft_per_sample_grads2['concatenated_weights'].mean(dim=0, keepdim=True)).T
#         scalars['rank_of_similarity']['concatenated_weights'] = self.matrix_rank(matrices['similarity']['concatenated_weights'], hermitian=False if x_true2 is not None else True)
#         return matrices, scalars
    
#     def compute_loss(self, params, buffers, sample, target):
#         batch = sample.unsqueeze(0)
#         targets = target.unsqueeze(0)
#         predictions = functional_call(self.model, (params, buffers), (batch,))
#         loss = self.criterion(predictions, targets)
#         return loss
    
#     def trace_of_cov(self, g):
#         g_mean = torch.mean(g, dim=0)
#         g = g - g_mean
#         tr = torch.mean(g.norm(dim=1)**2)
#         return tr.item()
    
#     def matrix_rank(self, g, hermitian=False):
#         return torch.linalg.matrix_rank(g, hermitian=hermitian)
    
            
# # wyliczyć sharpness dla macierzy podobieństwa, loader składa się z 500 przykładów
    
# class Stiffness(torch.nn.Module):
#     # add option to compute loss directly
#     # add option with train-val
#     def __init__(self, model, num_classes, x_true1, y_true1, logger=None, every_nb=5, x_true2=None, y_true2=None):
#         super().__init__()
#         self.per_sample_grad = PerSampleGrad(model)
#         self.num_classes = num_classes
#         idxs = torch.arange(0, x_true1.shape[0], every_nb)
#         self.x_true1 = x_true1[idxs]
#         self.y_true1 = y_true1[idxs]
#         self.x_true2 = x_true2[idxs] if x_true2 is not None else None
#         self.y_true2 = y_true2[idxs] if y_true2 is not None else None
#         self.logger = logger
        
#     def run_stiffness(self, step):
#         stifness_heatmaps = {}
#         stiffness_logs = {}
#         similarity_matrices, graham_matrices, cov_matrices, trace_of_cov_matrices, expected_stiffness_cosine, expected_stiffness_sign, \
#             c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine, c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign = self.forward(self.x_true1, self.y_true1, self.x_true2, self.y_true2)
        
#         stiffness_logs['stiffness_sharpness/similarity'] = self.sharpness(similarity_matrices['concatenated_weights'])
#         stiffness_logs['stiffness_sharpness/graham'] = self.sharpness(graham_matrices['concatenated_weights'])
#         stiffness_logs['stiffness_sharpness/cov'] = self.sharpness(cov_matrices['concatenated_weights'])
#         stiffness_logs['stiffness_sharpness/trace_of_cov'] = trace_of_cov_matrices['concatenated_weights']
        
#         fig1, ax1  = plt.subplots(1, 1, figsize=(10, 10))
#         fig2, ax2  = plt.subplots(1, 1, figsize=(10, 10))
#         fig3, ax3  = plt.subplots(1, 1, figsize=(10, 10))
#         stifness_heatmaps['stiffness/alignment_all'] = sns.heatmap(similarity_matrices['concatenated_weights'].data.cpu().numpy(), ax=ax1).get_figure()
#         # stifness_heatmaps['stiffness/graham_all'] = sns.heatmap(graham_matrices['concatenated_weights'].data.cpu().numpy(), annot=True).get_figure()
#         # stifness_heatmaps['stiffness/cov_all'] = sns.heatmap(cov_matrices['concatenated_weights'].data.cpu().numpy(), annot=True).get_figure()
#         stifness_heatmaps['stiffness/class_alignment_cosine'] = sns.heatmap(c_stiffness_cosine['concatenated_weights'].data.cpu().numpy(), ax=ax2).get_figure()
#         stifness_heatmaps['stiffness/class_alignment_sign'] = sns.heatmap(c_stiffness_sign['concatenated_weights'].data.cpu().numpy(), ax=ax3).get_figure()
        
#         stiffness_logs['stiffness/overall_stiffness_cosine'] = expected_stiffness_cosine['concatenated_weights']
#         stiffness_logs['stiffness/overall_stiffness_sign'] = expected_stiffness_sign['concatenated_weights']
#         stiffness_logs['stiffness/between_classes_cosine'] = stiffness_between_classes_cosine['concatenated_weights']
#         stiffness_logs['stiffness/within_classes_cosine'] = stiffness_within_classes_cosine['concatenated_weights']
#         stiffness_logs['stiffness/between_classes_sign'] = stiffness_between_classes_sign['concatenated_weights']
#         stiffness_logs['stiffness/within_classes_sign'] = stiffness_within_classes_sign['concatenated_weights']
        
#         stiffness_logs['steps/stiffness_train'] = step
        
#         self.logger.log_figures(stifness_heatmaps, step)
#         self.logger.log_scalars(stiffness_logs, step)
        
        
#     def forward(self, x_true, y_true, x_true2=None, y_true2=None):
#         similarity_matrices, graham_matrices, cov_matrices, trace_of_cov_matrices = self.per_sample_grad(x_true, y_true, x_true2, y_true2) # [<g_i/|g_i|, g_j/|g_j|>]_{i,j}, [<g_i, g_j>]_{i,j}, [<g_i-g, g_j-g>]_{i,j}
#         expected_stiffness_cosine = self.cosine_stiffness(similarity_matrices) 
#         expected_stiffness_sign = self.sign_stiffness(similarity_matrices)
#         # expected_stiffness_diagonal_cosine, expected_stiffness_diagonal_sign  = self.instance_stiffness(similarity_matrices, y_true, whether_sign=False)
#         c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine  = self.class_stiffness(similarity_matrices, y_true, whether_sign=False)
#         c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign  = self.class_stiffness(similarity_matrices, y_true, whether_sign=True)
#         return similarity_matrices, graham_matrices, cov_matrices, trace_of_cov_matrices, expected_stiffness_cosine, expected_stiffness_sign, c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine, c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign
    
#     def cosine_stiffness(self, similarity_matrices):
#         expected_stiffness = {k: torch.mean(similarity_matrix).item() for k, similarity_matrix in similarity_matrices.items()}
#         return expected_stiffness
    
#     def sign_stiffness(self, similarity_matrices):
#         expected_stiffness = {k: torch.mean(torch.sign(similarity_matrix)).item() for k, similarity_matrix in similarity_matrices.items()}
#         return expected_stiffness
    
#     def class_stiffness(self, similarity_matrices, y_true, whether_sign=False):
#         c_stiffness = {}
#         # extract the indices into dictionary from y_true tensor where the class is the same
#         indices = {i: torch.where(y_true == i)[0] for i in range(self.num_classes)}
#         indices = {k: v for k, v in indices.items() if v.shape[0] > 0}
#         for k, similarity_matrix in similarity_matrices.items():
#             c_stiffness[k] = torch.zeros((self.num_classes, self.num_classes), device=y_true.device)
#             for c1, idxs1 in indices.items():
#                 for c2, idxs2 in indices.items():
#                     sub_matrix = similarity_matrix[idxs1, :][:, idxs2]
#                     sub_matrix = torch.sign(sub_matrix) if whether_sign else sub_matrix
#                     c_stiffness[k][c1, c2] = torch.mean(sub_matrix) if c1 != c2 else (torch.sum(sub_matrix) - sub_matrix.size(0)) / (sub_matrix.size(0)**2 - sub_matrix.size(0))
                    
#         stiffness_between_classes = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in c_stiffness.items()}
#         stiffness_within_classes = {k: (torch.diagonal(v).sum() / v.size(0)).item() for k, v in c_stiffness.items()}
        
#         return c_stiffness, stiffness_between_classes, stiffness_within_classes  
        
#     def sharpness(self, similarity_matrix):
#         w, _ = torch.linalg.eig(similarity_matrix)
#         max_eig = torch.max(w.real) # .abs()??
#         return max_eig
    
    
    
##############################################################################################################

##############################################################################################################
    
    
import torch
from torch.func import functional_call, vmap, grad
from sklearn.cluster import SpectralClustering 
    
class PerSampleGrad(torch.nn.Module):
    # compute loss and grad per sample 
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss), in_dims=(None, None, 0, 0))
        
    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = self.criterion(predictions, targets)
        return loss

    def forward(self, x_true1, y_true1, x_true2=None, y_true2=None):
        matrices = {
            'similarity_11': {},
            'graham_11': {},
            'cov_11': {},
        }
        scalars = {
            'trace_of_cov_11': {},
            'rank_of_gradients_11': {},
            'rank_of_similarity_11': {},
            # 'rank_of_weights': {},
            'cumm_gradients_rank_11': {},
        }
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        
        # scalars['cumm_max_rank_of_weights'] = {k: min(v.shape)  for k, v in params.items() if 'weight' in k}
        # scalars['rank_of_weights'] = {k: self.matrix_rank(v) / min(v.shape)  for k, v in params.items() if 'weight' in k}
        # scalars['rank_of_weights']['concatenated_weights'] = sum(scalars['rank_of_weights'][tag] * scalars['cumm_max_rank_of_weights'][tag] for tag in scalars['rank_of_weights']) / sum(scalars[f'cumm_max_rank_of_weights'].values())
        # del scalars['cumm_max_rank_of_weights']
            
        ft_per_sample_grads1 = self.ft_criterion(params, buffers, x_true1, y_true1)
        ft_per_sample_grads1 = {k1: v.detach().data for k1, v in ft_per_sample_grads1.items()}
        concatenated_weights1 = torch.empty((x_true1.shape[0], 0), device=x_true1.device)
        if x_true2 is not None:
            matrices.update({
                    'similarity_22': {},
                    'graham_22': {},
                    'cov_22': {},
                    'similarity_12': {},
                    'graham_12': {},
                    'cov_12': {},
                })
            scalars.update({
                    'trace_of_cov_22': {},
                    'rank_of_gradients_22': {},
                    'rank_of_similarity_22': {},
                    'rank_of_similarity_12': {},
                    'cumm_gradients_rank_22': {},
                })
            ft_per_sample_grads2 = self.ft_criterion(params, buffers, x_true2, y_true2)
            ft_per_sample_grads2 = {k2: v.detach().data for k2, v in ft_per_sample_grads2.items()}
            concatenated_weights2 = torch.empty((x_true2.shape[0], 0), device=x_true2.device)
        
        for k in ft_per_sample_grads1:
            normed_ft_per_sample_grad1, concatenated_weights1 = self.prepare_variables(ft_per_sample_grads1, concatenated_weights1, scalars, tag=k, ind='11')
            if x_true2 is not None:
                normed_ft_per_sample_grad2, concatenated_weights2 = self.prepare_variables(ft_per_sample_grads2, concatenated_weights2, scalars, tag=k, ind='22')
                self.gather_metrics(ft_per_sample_grads2, ft_per_sample_grads2, normed_ft_per_sample_grad2, normed_ft_per_sample_grad2, matrices, scalars, tag=k, hermitian=True, ind='22')
                self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads2, normed_ft_per_sample_grad1, normed_ft_per_sample_grad2, matrices, scalars, tag=k, hermitian=False, ind='12')
            
            self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads1, normed_ft_per_sample_grad1, normed_ft_per_sample_grad1, matrices, scalars, tag=k, hermitian=True, ind='11')
            
        normed_concatenated_weights1 = self.prepare_concatenated_weights(ft_per_sample_grads1, concatenated_weights1, scalars, ind='11')
        if x_true2 is not None:
            normed_concatenated_weights2 = self.prepare_concatenated_weights(ft_per_sample_grads2, concatenated_weights2, scalars, ind='22')
            self.gather_metrics(ft_per_sample_grads2, ft_per_sample_grads2, normed_concatenated_weights2, normed_concatenated_weights2, matrices, scalars, tag='concatenated_weights', hermitian=True, ind='22')
            self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads2, normed_concatenated_weights1, normed_concatenated_weights2, matrices, scalars, tag='concatenated_weights', hermitian=False, ind='12')
        
        self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads1, normed_concatenated_weights1, normed_concatenated_weights1, matrices, scalars, tag='concatenated_weights', hermitian=True, ind='11')
        del scalars['cumm_gradients_rank_11']
        if x_true2 is not None:
            del scalars['cumm_gradients_rank_22']
        return matrices, scalars
    
    def trace_of_cov(self, g):
        g_mean = torch.mean(g, dim=0, keepdim=True)
        g -= g_mean
        tr = torch.mean(g.norm(dim=1)**2)
        return tr.item()
    
    def matrix_rank(self, g, hermitian=False):
        rank = np.linalg.matrix_rank(g.detach().data.cpu().numpy(), hermitian=hermitian).astype(float).mean()
        return rank
        # return torch.linalg.matrix_rank(g, hermitian=hermitian).float().mean().item()
    
    def prepare_variables(self, ft_per_sample_grads, concatenated_weights, scalars, tag, ind: str = None):
        if 'weight' in tag:
            scalars[f'cumm_gradients_rank_{ind}'][tag] = min(ft_per_sample_grads[tag].shape[1:])
            scalars[f'rank_of_gradients_{ind}'][tag] = self.matrix_rank(ft_per_sample_grads[tag]) / min(ft_per_sample_grads[tag].shape[1:]) # ???
        ft_per_sample_grads[tag] = ft_per_sample_grads[tag].reshape(ft_per_sample_grads[tag].shape[0], -1)
        normed_ft_per_sample_grad = ft_per_sample_grads[tag] / (1e-9 + torch.norm(ft_per_sample_grads[tag], dim=1, keepdim=True))
        concatenated_weights = torch.cat((concatenated_weights, ft_per_sample_grads[tag]), dim=1)
        scalars[f'trace_of_cov_{ind}'][tag] = self.trace_of_cov(ft_per_sample_grads[tag])
        return normed_ft_per_sample_grad, concatenated_weights
    
    def prepare_concatenated_weights(self, ft_per_sample_grads, concatenated_weights, scalars, ind: str = None):
        scalars[f'rank_of_gradients_{ind}']['concatenated_weights'] = sum(scalars[f'rank_of_gradients_{ind}'][tag] * scalars[f'cumm_gradients_rank_{ind}'][tag] for tag in scalars[f'rank_of_gradients_{ind}']) / sum(scalars[f'cumm_gradients_rank_{ind}'].values())
        ft_per_sample_grads['concatenated_weights'] = concatenated_weights
        normed_concatenated_weights = ft_per_sample_grads['concatenated_weights'] / (1e-9 + torch.norm(ft_per_sample_grads['concatenated_weights'], dim=1, keepdim=True))
        scalars[f'trace_of_cov_{ind}']['concatenated_weights'] = self.trace_of_cov(ft_per_sample_grads['concatenated_weights'])
        return normed_concatenated_weights
    
    def gather_metrics(self, ft_per_sample_grads1, ft_per_sample_grads2, normed_ft_per_sample_grad1, normed_ft_per_sample_grad2, matrices, scalars, tag, hermitian=False, ind: str = None):
        matrices[f'similarity_{ind}'][tag] = normed_ft_per_sample_grad1 @ normed_ft_per_sample_grad2.T
        matrices[f'graham_{ind}'][tag] = ft_per_sample_grads1[tag] @ ft_per_sample_grads2[tag].T / matrices[f'similarity_{ind}'][tag].shape[0]
        matrices[f'cov_{ind}'][tag] = (ft_per_sample_grads1[tag] - ft_per_sample_grads1[tag].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads2[tag] - ft_per_sample_grads2[tag].mean(dim=0, keepdim=True)).T / matrices[f'similarity_{ind}'][tag].shape[0]
        scalars[f'rank_of_similarity_{ind}'][tag] = self.matrix_rank(matrices[f'similarity_{ind}'][tag], hermitian=hermitian)
        if ind != '12':
            matrices[f'similarity_{ind}'][tag] = (matrices[f'similarity_{ind}'][tag] + matrices[f'similarity_{ind}'][tag].T) / 2
            matrices[f'graham_{ind}'][tag] = (matrices[f'graham_{ind}'][tag] + matrices[f'graham_{ind}'][tag].T) / 2
            matrices[f'cov_{ind}'][tag] = (matrices[f'cov_{ind}'][tag] + matrices[f'cov_{ind}'][tag].T) / 2
    
            
# wyliczyć sharpness dla macierzy podobieństwa, loader składa się z 500 przykładów
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
class Stiffness(torch.nn.Module):
    # add option to compute loss directly
    # add option with train-val
    def __init__(self, model, num_classes, x_true1, y_true1, logger=None, every_nb=1, x_true2=None, y_true2=None):
        super().__init__()
        self.per_sample_grad = PerSampleGrad(model)
        self.num_classes = num_classes
        idxs = torch.arange(0, x_true1.shape[0], every_nb)
        self.x_true1 = x_true1[idxs]
        self.y_true1 = y_true1[idxs]
        self.x_true2 = x_true2[idxs] if x_true2 is not None else None
        self.y_true2 = y_true2[idxs] if y_true2 is not None else None
        self.logger = logger
        
    def log_stiffness(self, step):
        stifness_heatmaps = {}
        stiffness_logs = {}
        stiffness_hists = {}
        matrices, scalars = self.forward(self.x_true1, self.y_true1, self.x_true2, self.y_true2)
        
        for tag in matrices[0]:
            stiffness_logs[f'stiffness_sharpness/{tag}'] = self.sharpness(matrices[0][tag]['concatenated_weights'])
            _, ax  = plt.subplots(1, 1, figsize=(10, 10))
            # set a heatmap pallette to red-white-blue
            stifness_heatmaps[f'stiffness/{tag}'] = sns.heatmap(matrices[0][tag]['concatenated_weights'].data.cpu().numpy(), ax=ax, center=0, cmap='PRGn').get_figure()
            if 'similarity' in tag and '12' not in tag:
                labels_true = self.y_true1.cpu().numpy() if '11' in tag else self.y_true2.cpu().numpy()
                labels_pred, unsolicited_ratio = self.clustering(matrices[0][tag]['concatenated_weights'], labels_true)
                acc = (labels_pred == labels_true).sum() / labels_true.shape[0]
                stiffness_logs[f'clustering/accuracy_{tag}'] = acc
                stiffness_logs[f'clustering/unsolicited_ratio_{tag}'] = unsolicited_ratio
                stiffness_hists[f'clustering/histogram_{tag}'] = labels_pred
                
        
        for tag in matrices[1]:
            _, ax  = plt.subplots(1, 1, figsize=(10, 10))
            stifness_heatmaps[f'class_stiffness/{tag}'] = sns.heatmap(matrices[1][tag]['concatenated_weights'].data.cpu().numpy(), ax=ax, center=0, cmap='PRGn').get_figure()
                   
        for tag in scalars[0]:
            stiffness_logs[f'traces of covs & ranks/{tag}'] = scalars[0][tag]['concatenated_weights']
            
        for tag in scalars[1]:
            stiffness_logs[f'stiffness/{tag}'] = scalars[1][tag]['concatenated_weights']
        
        stiffness_logs['steps/stiffness_train'] = step
        
        self.logger.log_figures(stifness_heatmaps, step)
        self.logger.log_scalars(stiffness_logs, step)
        self.logger.log_histogram(stiffness_hists, step)
        
        
    def forward(self, x_true1, y_true1, x_true2=None, y_true2=None):
        matrices = defaultdict(dict)
        scalars = defaultdict(dict)
        matrices[0], scalars[0] = self.per_sample_grad(x_true1, y_true1, x_true2, y_true2) # [<g_i/|g_i|, g_j/|g_j|>]_{i,j}, [<g_i, g_j>]_{i,j}, [<g_i-g, g_j-g>]_{i,j}
        scalars[1]['expected_stiffness_cosine_11'] = self.cosine_stiffness(matrices[0]['similarity_11']) 
        scalars[1]['expected_stiffness_sign_11'] = self.sign_stiffness(matrices[0]['similarity_11']) 
        matrices[1]['c_stiffness_cosine_11'], scalars[1]['stiffness_between_classes_cosine_11'], scalars[1]['stiffness_within_classes_cosine_11']  = self.class_stiffness(matrices[0]['similarity_11'], y_true1, whether_sign=False)
        matrices[1]['c_stiffness_sign_11'], scalars[1]['stiffness_between_classes_sign_11'], scalars[1]['stiffness_within_classes_sign_11']  = self.class_stiffness(matrices[0]['similarity_11'], y_true1, whether_sign=True)
        if x_true2 is not None:
            scalars[1]['expected_stiffness_cosine_12'] = self.cosine_stiffness(matrices[0]['similarity_12']) 
            scalars[1]['expected_stiffness_sign_12'] = self.sign_stiffness(matrices[0]['similarity_12']) 
            scalars[1]['expected_stiffness_cosine_22'] = self.cosine_stiffness(matrices[0]['similarity_22']) 
            scalars[1]['expected_stiffness_sign_22'] = self.sign_stiffness(matrices[0]['similarity_22']) 
            matrices[1]['c_stiffness_cosine_12'], scalars[1]['stiffness_between_classes_cosine_12'], scalars[1]['stiffness_within_classes_cosine_12']  = self.class_stiffness(matrices[0]['similarity_12'], y_true1, whether_sign=False)
            matrices[1]['c_stiffness_sign_12'], scalars[1]['stiffness_between_classes_sign_12'], scalars[1]['stiffness_within_classes_sign_12']  = self.class_stiffness(matrices[0]['similarity_12'], y_true1, whether_sign=True)
            matrices[1]['c_stiffness_cosine_22'], scalars[1]['stiffness_between_classes_cosine_22'], scalars[1]['stiffness_within_classes_cosine_22']  = self.class_stiffness(matrices[0]['similarity_22'], y_true1, whether_sign=False)
            matrices[1]['c_stiffness_sign_22'], scalars[1]['stiffness_between_classes_sign_22'], scalars[1]['stiffness_within_classes_sign_22']  = self.class_stiffness(matrices[0]['similarity_22'], y_true1, whether_sign=True)
            scalars[1]['expected_stiffness_diagonal_cosine_12'], scalars[1]['expected_stiffness_diagonal_sign_12'] = self.diagonal_instance_stiffness(matrices[0]['similarity_12'])
        return matrices, scalars
    
    def cosine_stiffness(self, similarity_matrices):
        expected_stiffness = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        return expected_stiffness
    
    def sign_stiffness(self, similarity_matrices):
        expected_stiffness = {k: ((torch.sum(torch.sign(v)) - torch.diagonal(torch.sign(v)).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        return expected_stiffness
    
    def class_stiffness(self, similarity_matrices, y_true, whether_sign=False):
        c_stiffness = {}
        # extract the indices into dictionary from y_true tensor where the class is the same
        indices = {i: torch.where(y_true == i)[0] for i in range(self.num_classes)}
        indices = {k: v for k, v in indices.items() if v.shape[0] > 0}
        for k, similarity_matrix in similarity_matrices.items():
            c_stiffness[k] = torch.zeros((self.num_classes, self.num_classes), device=y_true.device)
            for c1, idxs1 in indices.items():
                for c2, idxs2 in indices.items():
                    sub_matrix = similarity_matrix[idxs1, :][:, idxs2]
                    sub_matrix = torch.sign(sub_matrix) if whether_sign else sub_matrix
                    c_stiffness[k][c1, c2] = torch.mean(sub_matrix) if c1 != c2 else (torch.sum(sub_matrix) - sub_matrix.size(0)) / (sub_matrix.size(0)**2 - sub_matrix.size(0))
                    
        stiffness_between_classes = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in c_stiffness.items()}
        stiffness_within_classes = {k: (torch.diagonal(v).sum() / v.size(0)).item() for k, v in c_stiffness.items()}
        
        return c_stiffness, stiffness_between_classes, stiffness_within_classes  
    
    def diagonal_instance_stiffness(self, similarity_matrices):
        expected_diag_stiffness_cosine = {k: torch.mean(torch.diag(v)).item() for k, v in similarity_matrices.items()}
        expected_diag_stiffness_sign = {k: torch.mean(torch.sign(torch.diag(v))).item() for k, v in similarity_matrices.items()}
        return expected_diag_stiffness_cosine, expected_diag_stiffness_sign
        
    def sharpness(self, similarity_matrix):
        w, _ = torch.linalg.eig(similarity_matrix)
        max_eig = torch.max(w.real) # .abs()??
        return max_eig.item()
    
    def clustering(self, similarity_matrix, labels_true):
        similarity_matrix_ = similarity_matrix.cpu().numpy()
        labels_pred = SpectralClustering(n_clusters=self.num_classes, affinity='precomputed', n_init=100, assign_labels='discretize').fit_predict((1+similarity_matrix_)/2)
        labels_pred = self.retrieve_info(labels_pred, labels_true)
        return labels_pred
    
    
    def retrieve_info(self, cluster_labels, y_train):
        # Initializing
        unsolicited_ratio = 0.0
        denominator = 0.0
        reference_labels = {}
        # For loop to run through each label of cluster label
        for label in range(len(np.unique(y_train))):
            index = np.where(cluster_labels==label, 1, 0)
            dist = np.bincount(y_train[index==1])
            num = dist.argmax()
            unsolicited_ratio += (dist.sum() - dist.max())
            denominator += dist.sum()
            reference_labels[label] = num
        proper_labels = [reference_labels[label] for label in cluster_labels]
        proper_labels = np.array(proper_labels)
        unsolicited_ratio /= denominator
        return proper_labels, unsolicited_ratio
