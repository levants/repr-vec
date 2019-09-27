"""
Created on Sep 17, 2019

Generating Grad-CAM hit-maps

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fastai.vision import *
from fastai.vision.learner import hook_output

from repr.models.input_utils import prepare


def grad_cam(model, im, cl, preprocessors, heatmap_thresh: int = 16):
    """
    Hitmap the model
    Args:
        model: network model
        im: input image
        cl: class index
        preprocessors: preprocessors for image
        heatmap_thresh: hit-map threshold

    Returns:
        mult: hit-maped image
    """
    seq = nn.Sequential(*list(model.children()))
    m = seq.eval()
    cl = int(cl)
    xb = prepare(preprocessors, im)  # put into a minibatch of batch size = 1
    model_body = m[:-2]
    with hook_output(model_body) as hook_a:
        with hook_output(model_body, grad=True) as hook_g:
            preds = m(xb)
            preds[0, int(cl)].backward()
    acts = hook_a.stored[0].cpu()  # activation maps
    if (acts.shape[-1] * acts.shape[-2]) >= heatmap_thresh:
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = F.relu((acts * grad_chan[..., None, None]).sum(0))

        return mult
