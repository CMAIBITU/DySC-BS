'''
numpy 的增强，可以理解为在转换为tensor前的
'''

import random
import uuid
from typing import List, Union

import kl
import numpy as np
from scipy.fft import fftn, ifftn
from termcolor import colored
from scipy.interpolate import interp1d


class Transform(object):
    def __init__(self):
        pass
    
    def forward(self, x, **args):
        return x
    
    def __call__(self, x, **args) :
        return self.forward(x, **args)
    
    def draw_before_aug(self, ts, c=0, channelFirst=True):
        if channelFirst:
            y = ts[c,:]
        else:
            y = ts[:,c]
        kl.diag.plot_together(y, color='green', draw_point=False)
    
    def draw_after_aug(self, ts, c=0, channelFirst=True):
        if channelFirst:
            y = ts[c,:]
        else:
            y = ts[:,c]
        kl.diag.plot_together(y, color='red', draw_point=False)
        name =  kl.stuff.get_readable_time() + '_' + ''.join(str(uuid.uuid4()).split('-'))[:16]
        kl.diag.show(f'tmp/augmentation/{name}.jpg')
        
    def to_dict(self):
        d = self.__dict__
        result = {}
        for key in d:
            obj = d[key]
            if isinstance(obj, Transform):
                result[key] = obj.to_dict()
            else:
                result[key] = obj
        return {self.__class__.__name__: result}
        
  
class ConsistentArgsTransMixin(object):
    def new_transform(self, **args):
        raise NotImplementedError()
    
    
class Compose(Transform):
    def __init__(self, transfomrs: List[Transform], drawChannel=None, channelFirst=False):
        super().__init__()
        self.transforms = transfomrs
        self.drawChannel = drawChannel
        self.channelFirst = channelFirst
        
    def forward(self, x, **args):
        if self.drawChannel is not None:
            self.draw_before_aug(x, 
                                c=self.drawChannel, 
                                channelFirst=self.channelFirst)
        for f in self.transforms:
            x = f(x, **args)
            
        if self.drawChannel is not None:
            self.draw_after_aug(x, 
                                c=self.drawChannel, 
                                channelFirst=self.channelFirst)
        return x
    
    def to_dict(self):
        # d = self.__dict__
        # result = {}
        # for key in d:
        #     obj = d[key]
        #     if isinstance(obj, Transform):
        #         result[key] = obj.__dict__
        #     else:
        #         result[key] = obj
        # return result
        ls = [ t.to_dict() for t in self.transforms]
        return {'transforms': ls}
    

class RandomCrop(Transform, ConsistentArgsTransMixin):
    def __init__(self, l):
        super().__init__()
        self.l = l
    
    def forward(self, x:Union[np.ndarray, List[np.ndarray]], **args):
        """_summary_

        Args:
            x (L,F): 时间序列
        """
        if isinstance(x, List):
            assert len(x[0]) >= self.l
            start = random.randint(0, len(x[0]) - self.l - 1)
            return [ data[start: start + self.l] for data in x]
        else:
            assert len(x) >= self.l
            start = random.randint(0, len(x) - self.l - 1)
            return x[start: start+ self.l]
        
    def new_transform(self, l):
        assert l >= self.l
        start = random.randint(0, l - self.l - 1)
        return Crop(start, start+ self.l)
    
    '''
    之后改用ConsistentArgsTransMixin.new_transform
    '''
    def get_random_crop_func(self, l):
        assert l >= self.l
        start = random.randint(0, l - self.l - 1)
        return Crop(start, start+ self.l)
        #  return random.randint(0, l - self.l - 1)

def cropAndPadding(x:np.ndarray, l:int, p=0):
    '''
    x的第一维进行crop, 长度不够补充p
    '''
    if len(x) <  l:
        shape = list(x.shape)
        shape[0] = l
        y = np.zeros(shape, dtype=x.dtype)
        y[:] = p
        y[:len(x)] = x
    else:
        s = random.randint(0, len(x) - l)
        y = x[s: s+l]
    return y
        

class RandomCropMaxMin(Transform, ConsistentArgsTransMixin):
    """
    RandomCrop,但保证结果长度在min与max之间
    如果ts小于min, 则返回整个ts
    如果 ts小于 max, 则返回长度在 [min, len(ts)] 间

    Args:
        Transform (_type_): _description_
        ConsistentArgsTransMixin (_type_): _description_
    """
    def __init__(self, min_len, max_len=None):
        super().__init__()
        self.min_len = min_len
        self.max_len = max_len
    
    def forward(self, x:np.ndarray, **args):
        """_summary_

        Args:
            x (L,F): 时间序列
        """
        # if isinstance(x, List):
        #     assert len(x[0]) >= self.l
        #     start = random.randint(0, len(x[0]) - self.l - 1)
        #     return [ data[start: start + self.l] for data in x]
        # else:
        #     assert len(x) >= self.l
        #     start = random.randint(0, len(x) - self.l - 1)
        #     return x[start: start+ self.l]
        crop = self.new_transform(x.shape[0])
        return crop(x)
        
    def new_transform(self, ts_len):
        # assert l >= self.min_
        if self.max_len is None:
            max_len = ts_len
        else:
            max_len = min(ts_len, self.max_len)

        min_len = min(ts_len, self.min_len)
        
        # try:
        crop_l = random.randint(min_len, max_len)    
            
        start = random.randint(0, ts_len - crop_l)
        # except:
        #     pass
        return Crop(start, start + crop_l)
    
    '''
    之后改用ConsistentArgsTransMixin.new_transform
    '''
    def get_random_crop_func(self, l):
        assert l >= self.l
        start = random.randint(0, l - self.l - 1)
        return Crop(start, start+ self.l)
        #  return random.randint(0, l - self.l - 1)


class RandomCropAndPadding(Transform, ConsistentArgsTransMixin):
    def __init__(self, l, p=0):
        super().__init__()
        self.l = l
        self.p = p
    
    def forward(self, x:Union[np.ndarray, List[np.ndarray]], **args):
        """_summary_

        Args:
            x (L,F): 时间序列
        """
        if isinstance(x, List):
            return [ cropAndPadding(data, self.l, self.p) for data in x]
        else:
            return cropAndPadding(x, self.l, self.p)
        
    def new_transform(self, l):
        '''
        l 是传入 forward的x 的长度，真正的长度
        '''
        if l > self.l:
            start = random.randint(0, l - self.l)
        else:
            start = 0
        return CropAndPadding(start, start+ self.l, self.p)
    
    # '''
    # 之后改用ConsistentArgsTransMixin.new_transform
    # '''
    # def get_random_crop_func(self, l):
    #     if l > self.l:
    #         start = random.randint(0, l - self.l - 1)
    #     else:
    #         start = 0
    #     return CropAndPadding(start, start+ self.l, self.p)
    
class Standardization(Transform):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
    
    def forward(self, x, **args):
        if self.dim is None:
            return kl.data.get_stand(x)
        else:
            m = np.mean(x, axis=self.dim)
            std = np.std(x, axis=self.dim)
            if np.any(std == 0):
                std += 1e-10
                # print('ffffffff')
            # assert np.all(std != 0)
            # if std == 0:
            #     raise ZeroDivisionError()
            #     std = 1e-4
            x = (x - m) / std 
            return x
    
            
    
class Jitter(Transform):
    # https://arxiv.org/pdf/1706.00527.pdf
    def __init__(self, sigma=0.8):
        super().__init__()
        self.sigma=sigma
    def forward(self, x):
        a = np.random.normal(loc=0., scale=self.sigma, size=x.shape)
        return x + a
    
class Identity(Transform):
    def forward(self, x, **args):
        return super().forward(x, **args)
    
 
class Scaling(Transform):
    # https://arxiv.org/pdf/1706.00527.pdf
    def __init__(self, sigma=.2):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x, **args): # L, F
        # 同一时间点，每个channel,scale是一样的。要保持同一时间点, ROI之间的相关性不变
        factor = np.random.normal(loc=1., scale=self.sigma, size=x.shape[0])
        return x * factor[:,None]

class Permutation(Transform):
    def __init__(self, max_segments=5, seg_mode="random"):
        super().__init__()
        self.max_segments = max_segments
        self.seg_mode=seg_mode
    
    def forward(self, x, **args):
        num_segs = np.random.randint(1, self.max_segments)
        # num_segs = self.max_segments
        if num_segs == 1:
            return x
        orig_steps = np.arange(x.shape[0])
        # ret = np.zeros_like(x)
        if self.seg_mode == "random":
            split_points = np.random.choice(x.shape[0] - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        warp = np.concatenate(np.random.permutation(splits))
        return x[warp]


class Crop(Transform):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
        
    def forward(self, x, **args):
        return x[self.start: self.end]
    
class CropAndPadding(Transform):
    def __init__(self, start, end, p=0):
        super().__init__()
        self.start = start
        self.end = end
        self.p = p
        
    def forward(self, x, **args):
        if len(x) >= self.end:
            return x[self.start: self.end]
        else:
            shape = list(x.shape)
            shape[0] = self.end - self.start
            y = np.zeros(shape, dtype=x.dtype)
            y[:] = self.p
            y[:len(x)] = x
            return y
            
            
class AmplitudeFlip(Transform):
    def __init__(self, p=0.5, m=0):
        super().__init__()
        print(colored('vertical Flip will symmetry at m ', 'red'))
        self.p = p
        self.m = m
    
    def forward(self, x, **args):
        if random.random() >= self.p:
            return -x + 2 * self.m
        else:
            return x

class TimeFlip(Transform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x, **args):
        if random.random() >= self.p:
            return np.flip(x, axis=0)
        else:
            return x


class InterpTR2(Transform):
    def __init__(self,p=0.5, a=-1, b=1):
        super().__init__()
        self.p=p
        self.a=a
        self.b=b
    
    def forward(self, ts, **args):
        if random.random() >= self.p:
            x = np.arange(0, ts.shape[0] * 2, 2)
            # ts = ts.transpose()
            # quadratic
            func = interp1d(x, ts, 'cubic', axis=-2, fill_value="extrapolate")
            new_x = np.arange(0, ts.shape[0] * 2, 2) + random.uniform(self.a, self.b)
            y = func(new_x) 
            return y   
        else:
            return ts

# class LEiDA(Transform):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, ts, **args):
#         leida, phase, dFC = cal_leida(time_series[None, :, :])


def flip():
    pass


# ============== 频域下增强 ============
# 参考：https://www.cnblogs.com/LXP-Never/p/11558302.html
# 测试，源文件写在kl的pycharm，bj_tmp的tmp_augmentation.py

def get_magnitude_phase_from_ts(ts):
    """
    通过傅里叶变换，得到振幅与相位
    :param ts: shape: (C,L) 或者 L
    :return: 振幅，相位角（radius）
    """
    
    f = fftn(ts, axes=-1)
    # 得到振幅与距离
    magnitude = np.abs(f)
    phase = np.angle(f)
    return magnitude, phase

def ifft_from_magnitude_phase(magnitude, phase):
    f = (np.cos(phase) + np.sin(phase) * 1j) * magnitude
    return ifftn(f, axes=-1).real

def remove_freq(ts, fs=list(range(20,90-20))):
    ts = ts.T
    amp, phase = get_magnitude_phase_from_ts(ts)
    # for f in fs:
    #     amp[f] = 0
    amp[fs] = 0
    ts = ifft_from_magnitude_phase(amp, phase)
    return ts.T

def random_scale_magnitude_in_freq(ts, s, pertub_ratio):
    """
    对某些channel的某些频率进行scale。
    :param ts: shape: (C, L)
    :param s: 最大scale到原来的多少 b∈(1-s, 1+s)， scale到原来幅度的b倍
    :param pertub_ratio: ts的频域的freq，多大概率对振幅进行random_scale
    :return:
    """
    if s == 0 or pertub_ratio == 0:
        return ts
    m, p = get_magnitude_phase_from_ts(ts)
    mask = np.random.uniform(0, 1, size=ts.shape) < pertub_ratio
    scales = np.random.uniform(1-s, 1+s, size=ts.shape)
    m[mask] = m[mask] * scales[mask]
    ts = ifft_from_magnitude_phase(m, p)
    return ts


# RobustTAD 的振幅增强
def random_aug_magnitude_RobustTAD(ts, r, mK, qA=1):
    """
    对某个频域片段的振幅进行增强。增幅设置为v，v服从 正态U(mu, sigma * qA),
    mu，与sigma是这个片段频域的统计均值与标准差。qA是控制参数。
    频域片段定义为：i 到 i + l
    l = r * L / 2, 每个channel都转换到时域，每个个channel中取mK个index(i),
    譬如channel c对应频域freq，的振幅为m，m[c, i:i +l]
    :param ts: shape: (C, L)
    :param r: 控制片段的长度
    :param mK: 每个channel 取mK个片段
    :param qA:
    :return:
    """
    l = int(r * ts.shape[-1] / 2)
    m, p = get_magnitude_phase_from_ts(ts)
    for c in range(ts.shape[0]):
        indeics = np.random.choice(ts.shape[1] - l, mK, replace=False)
        for i in indeics:
            mu = np.mean(m[c, i:i+l])
            std = np.std(m[c, i:i+l])
            v = random.gauss(mu, qA * std)
            m[c, i:i + l] = v
    return ifft_from_magnitude_phase(m, p)

# RobustTAD 的相位增强
def random_add_phase_RobustTAD(ts, r, mK, sigma):
    """
    对某个频域片段的振幅进行增强。增幅设置为v，v服从 正态U(mu, sigma * qA),
    mu，与sigma是这个片段频域的统计均值与标准差。qA是控制参数。
    频域片段定义为：i 到 i + l
    l = r * L / 2, 每个channel都转换到时域，每个个channel中取mK个index(i),
    譬如channel c对应频域freq，的振幅为m，m[c, i:i +l]
    :param ts: shape: (C, L)
    :param r: 控制片段的长度
    :param mK: 每个channel 取mK个片段
    :param sigma: d ~ N(0, sigma^2)
    :return:
    """
    l = int(r * ts.shape[-1] / 2)
    m, p = get_magnitude_phase_from_ts(ts)
    for c in range(ts.shape[0]):
        indeics = np.random.choice(ts.shape[1] - l, mK, replace=False)
        for i in indeics:
            v = random.gauss(0, sigma)
            p[c, i:i + l] = p[c, i:i + l] + v
    return ifft_from_magnitude_phase(m, p)


class RandomScaleMagnitudeFreq(Transform):
    def __init__(self, s, pertub_ratio, channelFirst=True):
        """_summary_

        Args:
            s (float): 最大scale到原来的多少 b∈(1-s, 1+s)， scale到原来幅度的b倍
            pertub_ratio (float): ts的频域的freq，多大概率对振幅进行random_scale
        """
        super().__init__()
        self.s = s
        self.pertub_ratio = pertub_ratio
        self.channelFirst = channelFirst
        
    def forward(self, x, **args):
        '''
            x是 np.ndarray
        '''
        if self.channelFirst is False:
            x = x.T
        
        x = random_scale_magnitude_in_freq(x, self.s, self.pertub_ratio)
        
        if self.channelFirst is False:
            x = x.T
            
        return x
    
class RemoveChannel(Transform):
    def __init__(self, channel_list=None):
        super().__init__()
        self.channel_list = channel_list
    
    def forward(self, x, **args):
        if self.channel_list is not None:
            x[:, self.channel_list] = 0
        return x
    
# class ToFC(Transform):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, x, **args):
#         if self.channel_list is not None:
#             x[:, self.channel_list] = 0
#         return x
    
class TemporalInteralJump(Transform):
    def __init__(self, interal=1):
        super().__init__()
        self.interal = interal
    
    def forward(self, x, **args):
        return x[::self.interal]
    
    

class RandomRemoveChannel(Transform):
    def __init__(self, num_channel, num_remove):
        super().__init__()
        self.num_remove = num_remove
        self.channels = list(range(num_channel))
    
    def forward(self, x, **args):
        nc = random.randint(0, self.num_remove)
        cs = np.random.choice(self.channels, nc)
        x[:, cs] = 0
        return x


class RandomRemoveFreq(Transform):
    def __init__(self, num_channel, num_remove):
        super().__init__()
        self.num_remove = num_remove
        self.channels = list(range(1, num_channel))
    
    def forward(self, x, **args):
        nc = random.randint(0, self.num_remove)
        cs = np.random.choice(self.channels, nc)
        x = remove_freq(x, cs)
        return x
         

class RemoveFreq(Transform):
    def __init__(self, fs=list(range(20,90-20))):
        super().__init__()
        self.fs = fs
    
    def forward(self, x, **args):
        return remove_freq(x, self.fs)
    
class RandomAugMagnitude_RobustTAD(Transform):
    def __init__(self, r, mK, qA=1, channelFirst=True, ratio=1):
        """
            对某个频域片段的振幅进行增强。增幅设置为v，v服从 正态U(mu, sigma * qA),
            mu，与sigma是这个片段频域的统计均值与标准差。qA是控制参数。
            频域片段定义为：i 到 i + l
            l = r * L / 2, 每个channel都转换到时域，每个个channel中取mK个index(i),
            譬如channel c对应频域freq，的振幅为m，m[c, i:i +l]
            :param ts: shape: (C, L)
            :param r: 控制片段的长度
            :param mK: 每个channel 取mK个片段
            :param qA:
        """
        super().__init__()
        self.r = r
        self.mK = mK
        self.qA = qA
        self.channelFirst = channelFirst
        self.ratio=ratio
        
    def forward(self, x, **args):
        if random.random() >= self.ratio:
            return x
        
        if self.channelFirst is False:
            x = x.T
        
        x = random_aug_magnitude_RobustTAD(x, self.r, self.mK, self.qA)
        
        if self.channelFirst is False:
            x = x.T
        
        return x
            
class RandomAddPhase_RobustTAD(Transform):
    def __init__(self, r, mK, sigma, channelFirst=True, ratio=1):
        super().__init__()
        self.r = r
        self.mK = mK
        self.sigma = sigma
        self.channelFirst = channelFirst
        self.ratio = ratio
        
    def forward(self, x, **args):
        
        if random.random() >= self.ratio:
            return x
        
        if self.channelFirst is False:
            x = x.T
        
        x = random_add_phase_RobustTAD(x, self.r, self.mK, self.sigma)
        
        if self.channelFirst is False:
            x = x.T
        
        return x
    

# ============== 频域下增强 end============== 

# 一下是mvts_transformer的，论文："A Transformer-based Framework for Multivariate Time Series Representation Learning"
def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def noise_mask(L, C, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        L: seq_length
        C: feat_dim
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones((L, C), dtype=bool)
            for m in range(C):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(L, lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(L, lm, masking_ratio), 1), C)
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=(L,C), replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(L, 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), C)

    return mask


def test():
    # per = Permutation(5)
    # x = np.arange(18).reshape(9,2)
    # x = per(x)
    # print(x)
    c = Compose([RandomCropAndPadding(90), 
                 InterpTR2(p=0.0,a=-1, b=1),
                 RemoveChannel([]),
                 AmplitudeFlip(0.2,m=0),
                 Jitter(0.03),
                 Scaling(sigma=0.23),])
    
    print(c.to_dict())
    

if __name__ == '__main__':
    test()

