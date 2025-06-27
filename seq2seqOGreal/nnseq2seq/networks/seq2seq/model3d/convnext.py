import torch
import torch.nn as nn
import torch.nn.functional as F

from nnseq2seq.networks.seq2seq.hyperconv import hyperConv


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, D, W, H)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, padding_mode='zeros') # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, D, W, H) -> (N, D, W, H, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, D, W, H, C) -> (N, C, D, W, H)

        x = input + x
        return x


class hyperBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, style_dim, latent_dim=8, kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()
        #self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.pad = nn.ZeroPad3d(padding)#nn.ReflectionPad3d(padding)
        self.dwconv = hyperConv(style_dim, dim, dim, ksize=kernel_size, padding=0, groups=dim, weight_dim=latent_dim, ndims=3)
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.pwconv1 = hyperConv(style_dim, dim, dim*4, ksize=1, padding=0, weight_dim=latent_dim, ndims=3)
        self.act = nn.GELU()
        self.pwconv2 = hyperConv(style_dim, dim*4, dim, ksize=1, padding=0, weight_dim=latent_dim, ndims=3)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1,dim,1,1,1)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, s):
        input = x
        x = self.pad(x)
        x = self.dwconv(x, s)
        #x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x, s)
        x = self.act(x)
        x = self.pwconv2(x, s)
        
        if self.gamma is not None:
            x = self.gamma * x

        x = input + x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class ResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, n_layer, kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()

        self.n_layer = n_layer
        
        self.resblocks = nn.ModuleList()
        for _ in range(n_layer):
            self.resblocks.append(Block(dim, kernel_size=kernel_size, padding=padding, layer_scale_init_value=layer_scale_init_value))

    def forward(self, x):
        for res in self.resblocks:
            x = res(x)
        return x
    

class hyperResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, style_dim, n_layer, latent_dim=8, kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()

        self.n_layer = n_layer
        
        self.resblocks = nn.ModuleList()
        for _ in range(n_layer):
            self.resblocks.append(hyperBlock(dim, style_dim, latent_dim=latent_dim, kernel_size=kernel_size, padding=padding, layer_scale_init_value=layer_scale_init_value))

    def forward(self, x, s):
        for res in self.resblocks:
            x = res(x, s)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels, heads=1):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_heads = in_channels//heads

        self.norm = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')
        self.q = torch.nn.Conv3d(in_channels,
                                 self.dim_heads*heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 self.dim_heads*heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 self.dim_heads*heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(self.dim_heads*heads,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,d,w,h = q.shape
        q = q.reshape(b*self.heads,self.dim_heads,d*w*h)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b*self.heads,self.dim_heads,d*w*h) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b*self.heads,self.dim_heads,d*w*h)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,d,w,h)

        h_ = self.proj_out(h_)

        return x+h_
    

class CrossAttnBlock(nn.Module):
    def __init__(self, in_channels, in_channels_style, heads=1):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_heads = in_channels//heads

        self.norm = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Linear(in_channels_style, in_channels)
        self.v = torch.nn.Linear(in_channels_style, in_channels)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, s):
        h_ = x
        h_ = self.norm(h_)
        s_ = s
        q = self.q(h_)
        k = self.k(s_).permute(0,2,1)
        v = self.v(s_).permute(0,2,1)

        # compute attention
        b,c,d,w,h = q.shape
        sn = k.shape[-1]
        q = q.reshape(b*self.heads,self.dim_heads,d*w*h)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b*self.heads,self.dim_heads,sn) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(self.dim_heads)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b*self.heads,self.dim_heads,sn)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,d,w,h)

        h_ = self.proj_out(h_)

        return x+h_


class AttnResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, n_layer, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_attn=False):
        super().__init__()

        self.n_layer = n_layer
        self.use_attn = use_attn
        
        if use_attn:
            self.attnblocks = AttnBlock(dim)

        self.resblocks = nn.ModuleList()
        for _ in range(n_layer):
            self.resblocks.append(Block(dim, kernel_size=kernel_size, padding=padding, layer_scale_init_value=layer_scale_init_value))

    def forward(self, x):
        for i, res in enumerate(self.resblocks):
            if self.use_attn and i==self.n_layer//2:
                x = self.attnblocks(x)
            x = res(x)
        return x
    

class hyperAttnResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, style_dim, n_layer, latent_dim=8, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_attn=False):
        super().__init__()

        self.n_layer = n_layer
        self.use_attn = use_attn
        
        if use_attn:
            self.attnblocks = CrossAttnBlock(dim, style_dim)

        self.resblocks = nn.ModuleList()
        for _ in range(n_layer):
            self.resblocks.append(hyperBlock(dim, style_dim, latent_dim=latent_dim, kernel_size=kernel_size, padding=padding, layer_scale_init_value=layer_scale_init_value))

    def forward(self, x, s):
        for i, res in enumerate(self.resblocks):
            if self.use_attn and i==self.n_layer//2:
                x = self.attnblocks(x, s.unsqueeze(1))
            x = res(x, s)
        return x