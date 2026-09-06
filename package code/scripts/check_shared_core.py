"""Compare the public GPDR against classes read from untouched notebooks.

Short tests check kernel matrices, initialization, stochastic objectives,
gradients, Adam iterates, and density prediction. This is not a replacement
for the full-data, full-step figure/metric reproduction audit.
"""
import argparse
import ast
import math
from pathlib import Path
import sys
import types
from dataclasses import dataclass

import gpytorch
import numpy as np
import torch
from kmeans_pytorch import kmeans
from scipy.stats import t as student_t
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpdr import GPDR, fit_adam, prepare_toy_data, prepare_gini_data
from gpdr.paper import weather
from gpdr.paper.configuration import toy_model_inputs, weather_model_inputs, gini_model_inputs, predict_gini_row
from audit_support import verify_source


def reference_class(name, data, root):
    _, nb, _ = verify_source(root, name)
    module = types.ModuleType('reference_' + name)
    sys.modules[module.__name__] = module
    module.__dict__.update(data, torch=torch, nn=torch.nn, np=np, math=math,
                           gpytorch=gpytorch, dataclass=dataclass, kmeans=kmeans, dim_x=3)
    indices = {'toy': [0], 'weather': [14], 'gini': [18, 19]}[name]
    for index in indices:
        source = ''.join(nb['cells'][index]['source'])
        tree = ast.parse('\n'.join(l for l in source.splitlines() if not l.startswith('%')))
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                     and n.name in {'KernelParams', 'RBF_SE_ARD', 'chol_inv', 'LogisticGPMixtureDiag', 'BetaGPMixtureDiag'}]
        exec(compile(tree, str(root / f'reference_{name}'), 'exec'), module.__dict__)
    return module.__dict__['BetaGPMixtureDiag' if name == 'gini' else 'LogisticGPMixtureDiag']


def same(a, b, label):
    a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    b = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b):
        error = np.max(np.abs(a-b)) if a.shape == b.shape else 'shape'
        raise AssertionError(f'{label}: max difference {error}')


def check(name, root, device):
    torch.set_default_dtype(torch.float64 if name != 'weather' else torch.float32)
    if name == 'toy':
        data = prepare_toy_data(n_train=80, n_test=20)
        args = toy_model_inputs(data, m_induce=16)
        count = 16
    elif name == 'gini':
        data = prepare_gini_data(root)
        args = gini_model_inputs(data, m_induce=40)
        count = 40
    else:
        class LinearMean:
            def predict(self, x):
                return np.asarray(x) @ np.array([0.2, -0.1, 0.3])
        torch.manual_seed(10)
        data = dict(DF=3., SCALE_T=0.8, ORIGINAL_SIGMA_NORMAL=0.8/3,
                    gam=LinearMean(), math=math, student_t=student_t, torch=torch)
        weather.helpers(data)
        data['x_train'] = torch.rand(200, 3, device=device)
        data['y_train'] = torch.randn(200, device=device) * 0.8
        args = weather_model_inputs(data, m_induce=81)
        count = 81
    Ref = reference_class(name, data, root)
    rng_seed = 23
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    old = Ref(args['x'], args['y'], C=2, m_induce=count, beta=args['beta'])
    old = old.to(args['x'].device)
    torch.manual_seed(rng_seed); np.random.seed(rng_seed)
    new = GPDR(**args)
    for key in ('x','y','Vu','g','gy','gyy','K11uu','K11uu_inv','A','B','A2','mus','s_params','logits'):
        same(getattr(old,key), getattr(new,key), name+'/'+key)
    old_opt = torch.optim.Adam(old.parameters(), lr=0.5)
    new_opt = torch.optim.Adam(new.parameters(), lr=0.5)
    for step in range(8):
        values = []
        for model, opt in [(old,old_opt), (new,new_opt)]:
            opt.zero_grad(); torch.manual_seed(100+step)
            F, info = model.forward_objective(); F.backward()
            values.append((F.detach().clone(), {key: getattr(model,key).grad.detach().clone()
                                                for key in ('mus','s_params','logits')}))
            opt.step()
        same(values[0][0],values[1][0],f'{name}/step{step}/objective')
        for key in values[0][1]:
            same(values[0][1][key],values[1][1][key],f'{name}/step{step}/gradient/{key}')
        for key in ('mus','s_params','logits'):
            same(getattr(old,key), getattr(new,key), f'{name}/step{step}/{key}')
        print(name, 'step', step, 'exact', flush=True)
    # Exercise the actual public trainer against the notebook's update/averaging
    # convention, starting from identical parameters and RNG state.
    steps, window = 12, (0 if name == 'gini' else 3)
    before = name == 'toy'
    old_opt = torch.optim.Adam(old.parameters(),lr=0.5)
    reference_trace = {'F': [], 'w': [], 's_component_mean': []}
    average, count_average = None, 0
    torch.manual_seed(901)
    for step in range(steps):
        old_opt.zero_grad()
        F, info = old.forward_objective()
        if before:
            reference_trace['w'].append(torch.softmax(old.logits,0).detach().cpu().numpy().copy())
            reference_trace['s_component_mean'].append(old.s_params.mean(1).detach().cpu().numpy().copy())
        F.backward(); old_opt.step()
        if window and step >= steps-window:
            with torch.no_grad():
                if average is None:
                    average = {key: (param.clone() if before else torch.zeros_like(param))
                               for key,param in old.named_parameters()}
                    if before:
                        count_average = 1
                elif before:
                    count_average += 1
                    for key,param in old.named_parameters():
                        average[key] = (average[key]*(count_average-1)+param)/count_average
                if not before:
                    for key,param in old.named_parameters():
                        average[key] += param.data
                    count_average += 1
        if not before:
            reference_trace['w'].append(torch.softmax(old.logits,0).detach().cpu().numpy().copy())
            reference_trace['s_component_mean'].append(old.s_params.mean(1).detach().cpu().numpy().copy())
        reference_trace['F'].append(float(F.detach()))
    if average is not None:
        with torch.no_grad():
            for key,param in old.named_parameters():
                param.copy_(average[key] if before else average[key]/count_average)
    torch.manual_seed(901)
    new, trace = fit_adam(new,steps=steps,lr=0.5,average_last=window,
                          averaging='running' if before else 'sum',trace_at='before' if before else 'after')
    for key in reference_trace:
        same(reference_trace[key],trace[key],name+'/fit_adam/trace/'+key)
    for key in ('mus','s_params','logits'):
        same(getattr(old,key),getattr(new,key),name+'/fit_adam/'+key)
    if name == 'gini':
        row=data['df_test'].iloc[0]; Xrow=data['X_test'].iloc[0]
        a=old.predict_density(df_row=row, X_row=Xrow, M=151)
        b=predict_gini_row(new,data,df_row=row,X_row=Xrow,M=151)
        b=(b.y,b.reference_density,b.density,b.correction,b.pit)
    else:
        xstar=0.3 if name=='toy' else np.array([0.3,0.4,0.5])
        a=old.predict_density(xstar,-1.,2.,M=151)
        b=new.predict_density(xstar,-1.,2.,M=151)
        b=(b.y,b.base_density,b.density,b.correction,b.pit) if name=='toy' else b
    for i,(aa,bb) in enumerate(zip(a,b)):
        same(aa,bb,f'{name}/prediction/{i}')
    if name == 'toy':
        for i,(a,b) in enumerate(zip(old.predict_f_and_derivs(0.3,M=151), new.predict_f_and_derivs(0.3,M=151))):
            same(a,b,f'{name}/derivatives/{i}')
    print(name, 'PUBLIC CORE EXACT', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--example',choices=['toy','gini','weather'],required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    check(args.example,args.project_root,torch.device(args.device))
