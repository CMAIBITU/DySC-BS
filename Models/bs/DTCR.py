import torch
from torch import nn

class KMeansLossHelper(nn.Module):
    '''
        根据论文 Learning Representations for Time Series Clustering
        的k-means loss。论文的做法是H(即希望更新的那个特征表示) 和 F分开更新,具体实现是每9轮更新一次F,
        我们这里就是每10轮保存一次就算F的model,然后每次都用这个model计算F，不像官方代码那样保存F。
        所以具体应用时，需要继承这个类，实现cal_F方法。
        
    '''
    def __init__(self, num_cluster, bz=64):
        super().__init__()
        self.num_cluster = num_cluster
        self.init_F = torch.empty(bz, num_cluster)
        nn.init.trunc_normal_(self.init_F)
        self.init_F = self.init_F.cuda()
    
    def cal_F(self, x):
        """返回h对应的F

        Args:
            x (_type_): 
        return F: shape: N, num_cluster
        """
        pass
    
    def forward(self,x:torch.Tensor, h:torch.Tensor):
        # h: N,C， 注意，论文里面的H是 C，N, 同时，我们在原来的loss基础上，加上求平均，就是除以 N
        H = h.T
        F = self.cal_F(x) 
        if F is None:
            F = self.init_F[:h.shape[0]]
            return 0
        HF = torch.matmul(H, F)
        return (torch.trace(torch.matmul(h,H)) - torch.trace(torch.matmul(HF.T, HF))) / h.shape[0]
    

        
class KMeansLossHelperWithEncoder(KMeansLossHelper):
    """
        encoder 可以直接返回特征表示！
    Args:
        KMeansLossHelper (_type_): _description_
    """
    def __init__(self, num_cluster, bz):
        super().__init__(num_cluster, bz)
        self.encoder = None
    
    # def update_encoder(self, encoder):        
    #     self.encoder = encoder
    
    def cal_F(self, x:torch.Tensor):
        if self.encoder is None:
            # return self.init_F[:x.shape[0]]
            return None
        with torch.no_grad():
            h_ = self.encoder(x) # h_ 的shape必须是 N，C
            U, S, Vh = torch.linalg.svd(h_ ,full_matrices=True)
            # print('cal u......................')
        return U[:, :self.num_cluster]# 有怀疑，见testSVD
    

def testSVD():
    A = torch.randn((64, 200))
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    i = 0
    for i in range(32):
        print(torch.all(A @ Vh[i].T - S[i] * U[:,i] < 1e-5))
    # print(A @ Vh[:, i].T - S[i] * U[i])
    

if __name__ == "__main__":
    testSVD()