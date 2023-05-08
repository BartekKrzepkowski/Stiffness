from copy import deepcopy

import seaborn as sns

import torch
from pyhessian import hessian


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

class PerSampleGrad(torch.nn.Module):
    # compute loss and grad per sample 
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss), in_dims=(None, None, 0, 0))

    def forward(self, x_true, y_true):
        similarity_matrixes = {}
        graham_matrixes = {}
        cov_matrixes = {}
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        ft_per_sample_grads = self.ft_criterion(params, buffers, x_true, y_true)
        ft_per_sample_grads = {k: v.detach().data.cpu() for k, v in ft_per_sample_grads.items()}
        concatenated_weights = torch.empty((x_true.shape[0], 0))
        for k in ft_per_sample_grads:
            ft_per_sample_grads[k] = ft_per_sample_grads[k].reshape(x_true.shape[0], -1)
            concatenated_weights = torch.cat((concatenated_weights, ft_per_sample_grads[k]), dim=1)
            normed_ft_per_sample_grad = ft_per_sample_grads[k] / torch.norm(ft_per_sample_grads[k], dim=1, keepdim=True)
            similarity_matrixes[k] = normed_ft_per_sample_grad @ normed_ft_per_sample_grad.T
            graham_matrixes[k] = ft_per_sample_grads[k] @ ft_per_sample_grads[k].T
            cov_matrixes[k] = (ft_per_sample_grads[k] - ft_per_sample_grads[k].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads[k] - ft_per_sample_grads[k].mean(dim=0, keepdim=True)).T
        ft_per_sample_grads['concatenated_weights'] = concatenated_weights
        normed_concatenated_weights = ft_per_sample_grads['concatenated_weights'] / torch.norm(ft_per_sample_grads['concatenated_weights'], dim=1, keepdim=True)
        similarity_matrixes['concatenated_weights'] = normed_concatenated_weights @ normed_concatenated_weights.T
        graham_matrixes['concatenated_weights'] = ft_per_sample_grads['concatenated_weights'] @ ft_per_sample_grads['concatenated_weights'].T
        cov_matrixes['concatenated_weights'] = (ft_per_sample_grads['concatenated_weights'] - ft_per_sample_grads['concatenated_weights'].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads['concatenated_weights'] - ft_per_sample_grads['concatenated_weights'].mean(dim=0, keepdim=True)).T
        return similarity_matrixes, graham_matrixes, cov_matrixes
    
    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = self.criterion(predictions, targets)
        return loss
    
            
# wyliczyć sharpness dla macierzy podobieństwa, loader składa się z 500 przykładów
    
class Stiffness(torch.nn.Module):
    # add option to compute loss directly
    # add option with train-val
    def __init__(self, model, num_classes, x_data, y_data, logger=None):
        super().__init__()
        self.per_sample_grad = PerSampleGrad(model)
        self.num_classes = num_classes
        self.x_data = x_data
        self.y_data = y_data
        self.logger = logger
        
    def run_stiffness(self, step):
        stifness_heatmaps = {}
        stiffness_logs = {}
        similarity_matrixes, graham_matrixes, cov_matrixes, expected_stiffness_cosine, expected_stiffness_sign, c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine, c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign = self.forward(self.x_data, self.y_data)
        
        stiffness_logs['stiffness_sharpness/similarity'] = self.sharpness(similarity_matrixes['concatenated_weights'])
        stiffness_logs['stiffness_sharpness/graham'] = self.sharpness(graham_matrixes['concatenated_weights'])
        stiffness_logs['stiffness_sharpness/cov'] = self.sharpness(cov_matrixes['concatenated_weights'])
        
        # stifness_heatmaps['stiffness/alignment_all'] = sns.heatmap(similarity_matrixes['concatenated_weights'].data.cpu().numpy()).get_figure()
        # stifness_heatmaps['stiffness/graham_all'] = sns.heatmap(graham_matrixes['concatenated_weights'].data.cpu().numpy(), annot=True).get_figure()
        # stifness_heatmaps['stiffness/cov_all'] = sns.heatmap(cov_matrixes['concatenated_weights'].data.cpu().numpy(), annot=True).get_figure()
        stifness_heatmaps['stiffness/class_alignment_cosine'] = sns.heatmap(c_stiffness_cosine['concatenated_weights'].data.cpu().numpy()).get_figure()
        stifness_heatmaps['stiffness/class_alignment_sign'] = sns.heatmap(c_stiffness_sign['concatenated_weights'].data.cpu().numpy()).get_figure()
        
        stiffness_logs['stiffness/overall_stiffness_cosine'] = expected_stiffness_cosine['concatenated_weights']
        stiffness_logs['stiffness/overall_stiffness_sign'] = expected_stiffness_sign['concatenated_weights']
        stiffness_logs['stiffness/between_classes_cosine'] = stiffness_between_classes_cosine['concatenated_weights']
        stiffness_logs['stiffness/within_classes_cosine'] = stiffness_within_classes_cosine['concatenated_weights']
        stiffness_logs['stiffness/between_classes_sign'] = stiffness_between_classes_sign['concatenated_weights']
        stiffness_logs['stiffness/within_classes_sign'] = stiffness_within_classes_sign['concatenated_weights']
        
        stiffness_logs['steps/stiffness_train'] = step
        
        self.logger.log_figures(stifness_heatmaps, step)
        self.logger.log_scalars(stiffness_logs, step)
        
        
    def forward(self, x_true, y_true):
        similarity_matrixes, graham_matrixes, cov_matrixes = self.per_sample_grad(x_true, y_true) # [<g_i/|g_i|, g_j/|g_j|>]_{i,j}, [<g_i, g_j>]_{i,j}, [<g_i-g, g_j-g>]_{i,j}
        expected_stiffness_cosine = self.cosine_stiffness(similarity_matrixes) 
        expected_stiffness_sign = self.sign_stiffness(similarity_matrixes)
        c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine  = self.class_stiffness(similarity_matrixes, y_true, whether_sign=False)
        c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign  = self.class_stiffness(similarity_matrixes, y_true, whether_sign=True)
        return similarity_matrixes, graham_matrixes, cov_matrixes, expected_stiffness_cosine, expected_stiffness_sign, c_stiffness_cosine, stiffness_between_classes_cosine, stiffness_within_classes_cosine, c_stiffness_sign, stiffness_between_classes_sign, stiffness_within_classes_sign
    
    def cosine_stiffness(self, similarity_matrixes):
        expected_stiffness = {k: torch.mean(similarity_matrix).item() for k, similarity_matrix in similarity_matrixes.items()}
        return expected_stiffness
    
    def sign_stiffness(self, similarity_matrixes):
        expected_stiffness = {k: torch.mean(torch.sign(similarity_matrix)).item() for k, similarity_matrix in similarity_matrixes.items()}
        return expected_stiffness
    
    def class_stiffness(self, similarity_matrixes, y_true, whether_sign=False):
        c_stiffness = {}
        # extract the indices into dictionary from y_true tensor where the class is the same
        indices = {i: torch.where(y_true == i)[0] for i in range(self.num_classes)}
        indices = {k: v for k, v in indices.items() if v.shape[0] > 0}
        for k, similarity_matrix in similarity_matrixes.items():
            c_stiffness[k] = torch.zeros((self.num_classes, self.num_classes), device=y_true.device)
            for c1, idxs1 in indices.items():
                for c2, idxs2 in indices.items():
                    sub_matrix = similarity_matrix[idxs1, :][:, idxs2]
                    sub_matrix = torch.sign(sub_matrix) if whether_sign else sub_matrix
                    c_stiffness[k][c1, c2] = torch.mean(sub_matrix) if c1 != c2 else (torch.sum(sub_matrix) - sub_matrix.size(0)) / (sub_matrix.size(0)**2 - sub_matrix.size(0))
                    
        stiffness_between_classes = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in c_stiffness.items()}
        stiffness_within_classes = {k: (torch.diagonal(v).sum() / v.size(0)).item() for k, v in c_stiffness.items()}
        
        return c_stiffness, stiffness_between_classes, stiffness_within_classes  
        
    def sharpness(self, similarity_matrix):
        w, _ = torch.linalg.eig(similarity_matrix)
        max_eig = torch.max(w.real) # .abs()??
        return max_eig
    
    
    
