#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"../torchxrayvision/")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage
import skimage.io
import pprint
import json
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import skimage, skimage.filters
import torchxrayvision as xrv
import pickle

thispath = os.path.dirname(os.path.realpath(__file__))

class PneumoniaSeverityNetCIDC(torch.nn.Module):
    """
    Predicting COVID-19 Pneumonia Severity on Chest X-ray with Deep Learning
    https://arxiv.org/abs/2005.11856
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7451075/
    """
    def __init__(self):
        super(PneumoniaSeverityNetCIDC, self).__init__()
        self.model = xrv.models.DenseNet(weights="all")
        self.model.op_threshs = None
        self.theta_bias_geographic_extent = torch.from_numpy(np.asarray((0.8705248236656189, 3.4137437)))
        self.theta_bias_opacity = torch.from_numpy(np.asarray((0.5484423041343689, 2.5535977)))

    def forward(self, x):
        
        ret = {}
        feats = self.model.features(x)
        feats = F.relu(feats, inplace=True)
        ret["feats"] = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        ret["preds"] = self.model.classifier(ret["feats"])
        
        pred = ret["preds"][0,xrv.datasets.default_pathologies.index("Lung Opacity")]
        geographic_extent = pred*self.theta_bias_geographic_extent[0]+self.theta_bias_geographic_extent[1]
        opacity = pred*self.theta_bias_opacity[0]+self.theta_bias_opacity[1]
        
        ret["geographic_extent"] = torch.clamp(geographic_extent,0,8)
        ret["opacity"] = torch.clamp(opacity/6*8,0,8) #scaled to match new RALO score
        
        return ret
    
class PneumoniaSeverityNetStonyBrook(torch.nn.Module):
    """
    Trained on Stony Brook data https://doi.org/10.5281/zenodo.4633999
    Radiographic Assessment of Lung Opacity Score Dataset
    """
    def __init__(self):
        super(PneumoniaSeverityNetStonyBrook, self).__init__()
        self.basemodel = xrv.models.DenseNet(weights="all")
        self.basemodel.op_threshs = None
        self.net_geo = self.create_network(os.path.join(thispath, "weights/mlp_geo.pkl"))
        self.net_opa = self.create_network(os.path.join(thispath, "weights/mlp_opa.pkl"))        

    def create_network(self, path):
        coefs, intercepts = pickle.load(open(path,"br"))
        layers = []
        for i, (coef, intercept) in enumerate(zip(coefs, intercepts)):
            la = torch.nn.Linear(*coef.shape)
            la.weight.data = torch.from_numpy(coef).T.float()
            la.bias.data = torch.from_numpy(intercept).T.float()
            layers.append(la)
            if i < len(coefs)-1: # if not last layer
                layers.append(torch.nn.ReLU())
                
        return torch.nn.Sequential(*layers)
        
    def forward(self, x):
        
        ret = {}
        ret["feats"] = self.basemodel.features2(x)
        
        ret["geographic_extent"] = self.net_geo(ret["feats"])[0]
        ret["geographic_extent"] = torch.clamp(ret["geographic_extent"],0,8)
        
        ret["opacity"] = self.net_opa(ret["feats"])[0]
        ret["opacity"] = torch.clamp(ret["opacity"],0,8)
        
        return ret

def process(model, img_path, cuda=False):
    
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)  

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]                    
    
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])

    img = transform(img)
    
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda:
            img = img.cuda()
            model = model.cuda()

        outputs = model(img)
    
    outputs["img"] = img
    return outputs
    
def full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    parser.add_argument('-model_version', type=int, default=2, help='Version 1 or 2')
    parser.add_argument('-cuda', default=False, help='If cuda should be used or not', action='store_true')
    parser.add_argument('-batch', default=False, help='Batch mode to output for a csv file', action='store_true')
    parser.add_argument('-saliency_path', default=None, help='path to write the saliancy map as an image')

    cfg = parser.parse_args()

    if cfg.model_version == 1:
        model = PneumoniaSeverityNetCIDC()
    elif cfg.model_version == 2:
        model = PneumoniaSeverityNetStonyBrook()
    else:
        raise Exception("No model with version ", cfg.model_version)


    outputs = process(model, cfg.img_path, cfg.cuda)
    geo = float(outputs["geographic_extent"].cpu().numpy())
    opa = float(outputs["opacity"].cpu().numpy())
    
    if cfg.batch:
        to_output = {}
        to_output["img_path"] = cfg.img_path
        to_output["model"] = model.__class__.__name__
        to_output["geographic_extent"] = geo
        to_output["opacity"] = opa
        
        print(json.dumps(to_output))
    else:
        print("Predicting using model version {} ({})". format(cfg.model_version, model.__class__.__name__))
        print("Geographic Extent (0-8): {:1.4}".format(geo))
        print("Opacity (0-8): {:1.4}".format(opa))

    if cfg.saliency_path: 
        img = outputs["img"]

        img = img.requires_grad_()
        if cfg.cuda:
            model = model.cuda()
            img = img.cuda()
        outputs = model(img)
        grads = torch.autograd.grad(outputs["geographic_extent"], img)[0][0][0]
        blurred = skimage.filters.gaussian(grads.detach().cpu().numpy()**2, sigma=(5, 5), truncate=3.5)

        full_frame()
        my_dpi = 100
        fig = plt.figure(frameon=False, figsize=(224/my_dpi, 224/my_dpi), dpi=my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img[0][0].detach().cpu().numpy(), cmap="gray", aspect='auto')
        ax.imshow(blurred, alpha=0.5);
        plt.savefig(cfg.saliency_path)

