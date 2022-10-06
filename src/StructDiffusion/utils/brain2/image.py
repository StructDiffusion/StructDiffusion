"""
By Chris Paxton.

Copyright (c) 2018, Johns Hopkins University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Johns Hopkins University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL JOHNS HOPKINS UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import io
from PIL import Image

def GetJpeg(img):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(np.uint8(img))
    output = io.BytesIO()
    im.save(output, format="JPEG", quality=80)
    return output.getvalue()

def JpegToNumpy(jpeg):
    stream = io.BytesIO(jpeg)
    im = Image.open(stream)
    return np.asarray(im, dtype=np.uint8)

def ConvertJpegListToNumpy(data):
    length = len(data)
    imgs = []
    for raw in data:
        imgs.append(JpegToNumpy(raw))
    arr = np.array(imgs)
    return arr

def DepthToZBuffer(img, z_near, z_far):
    real_depth = z_near * z_far / (z_far - img * (z_far - z_near))
    return real_depth

def ZBufferToRGB(img, z_near, z_far):
    real_depth = z_near * z_far / (z_far - img * (z_far - z_near))
    depth_m = np.uint8(real_depth)
    depth_cm = np.uint8((real_depth-depth_m)*100)
    depth_tmm = np.uint8((real_depth-depth_m-0.01*depth_cm)*10000)
    return np.dstack([depth_m, depth_cm, depth_tmm])

def RGBToDepth(img, min_dist=0., max_dist=2.,):
    return (img[:,:,0]+.01*img[:,:,1]+.0001*img[:,:,2]).clip(min_dist, max_dist)
    #return img[:,:,0]+.01*img[:,:,1]+.0001*img[:,:,2]

def MaskToRGBA(img):
    buf = img.astype(np.int32)
    A = buf.astype(np.uint8)
    buf = np.right_shift(buf, 8)
    B = buf.astype(np.uint8)
    buf = np.right_shift(buf, 8)
    G = buf.astype(np.uint8)
    buf = np.right_shift(buf, 8)
    R = buf.astype(np.uint8)

    dims = [np.expand_dims(d, -1) for d in [R,G,B,A]]
    return np.concatenate(dims, axis=-1)

def RGBAToMask(img):
    mask = np.zeros(img.shape[:-1], dtype=np.int32)
    buf = img.astype(np.int32)
    for i, dim in enumerate([3,2,1,0]):
        shift = 8*i
        #print(i, dim, shift, buf[0,0,dim], np.left_shift(buf[0,0,dim], shift))
        mask += np.left_shift(buf[:,:, dim], shift)
    return mask

def RGBAArrayToMasks(img):
    mask = np.zeros(img.shape[:-1], dtype=np.int32)
    buf = img.astype(np.int32)
    for i, dim in enumerate([3,2,1,0]):
        shift = 8*i
        mask += np.left_shift(buf[:,:,:, dim], shift)
    return mask

def GetPNG(img):
    '''
    Save a numpy array as a PNG, then get it out as a binary blob
    '''
    im = Image.fromarray(np.uint8(img))
    output = io.BytesIO()
    im.save(output, format="PNG")#, quality=80)
    return output.getvalue()

def PNGToNumpy(png):
    stream = io.BytesIO(png)
    im = Image.open(stream)
    return np.array(im, dtype=np.uint8)

def ConvertPNGListToNumpy(data):
    length = len(data)
    imgs = []
    for raw in data:
        imgs.append(PNGToNumpy(raw))
    arr = np.array(imgs)
    return arr

def ConvertDepthPNGListToNumpy(data):
    length = len(data)
    imgs = []
    for raw in data:
        imgs.append(RGBToDepth(PNGToNumpy(raw)))
    arr = np.array(imgs)
    return arr

import cv2
def Shrink(img, nw=64):
    h,w = img.shape[:2]
    ratio = float(nw) / w
    nh = int(ratio * h)
    img2 = cv2.resize(img, dsize=(nw, nh),
        interpolation=cv2.INTER_NEAREST)
    return img2

def ShrinkSmooth(img, nw=64):
    h,w = img.shape[:2]
    ratio = float(nw) / w
    nh = int(ratio * h)
    img2 = cv2.resize(img, dsize=(nw, nh),
        interpolation=cv2.INTER_LINEAR)
    return img2

def CropCenter(img, cropx, cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = (x // 2) - (cropx // 2)
    starty = (y // 2) - (cropy // 2)
    return img[starty: starty + cropy, startx : startx + cropx]

