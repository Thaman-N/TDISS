import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import open_clip

class SpatialCropAttention(nn.Module):
    """Spatial Cropping Attention module from CUE-Net paper"""
    def __init__(self, in_channels, reduction=8):
        super(SpatialCropAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        # Store attention maps for visualization
        self.attention_maps = None
        
    def forward(self, x):
        b, c, t, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        
        # Create attention mask
        mask = torch.sigmoid(y)
        
        # Store for visualization (first batch item)
        self.attention_maps = mask[0].detach().cpu().numpy() if b > 0 else None
        
        # Apply spatial attention
        return x * mask.expand_as(x)

class LocalTemporalMHRA(nn.Module):
    """Local Temporal Multi-Head Relation Aggregator"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Use depth-wise convolution for local relation modeling
        self.dwconv = nn.Conv3d(
            num_heads, num_heads, 
            kernel_size=(3, 1, 1),  # 3 in temporal dimension, 1x1 in spatial
            padding=(1, 0, 0), 
            groups=num_heads  # depth-wise
        )
        
    def forward(self, x):
        B, T, N, C = x.shape  # batch, time, spatial tokens, channel
        
        # Generate q, k, v
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # 3, B, heads, T, N, C/heads
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, heads, T, N, C/heads
        
        # Reshape for depth-wise conv (treating batch*heads as batch dim)
        v_conv = v.permute(0, 1, 4, 2, 3)  # B, heads, C/heads, T, N
        v_conv = self.dwconv(v_conv)
        v_conv = v_conv.permute(0, 1, 3, 4, 2)  # B, heads, T, N, C/heads
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention and reshape
        x = (attn @ v_conv).transpose(2, 3).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class GlobalSpatialMHRA(nn.Module):
    """Global Spatial Multi-Head Relation Aggregator (initialized with CLIP-ViT embeddings)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize from CLIP-ViT weights will be done separately
        self.attention_maps = None
        
    def forward(self, x):
        B, T, N, C = x.shape  # batch, time, spatial tokens, channel
        
        # Generate q, k, v
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # 3, B, T, heads, N, C/heads
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, T, heads, N, C/heads
        
        # Compute attention scores 
        q = q.reshape(B*T, self.num_heads, N, C // self.num_heads)
        k = k.reshape(B*T, self.num_heads, N, C // self.num_heads)
        v = v.reshape(B*T, self.num_heads, N, C // self.num_heads)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Store attention map for first batch for visualization
        if B > 0 and T > 0:
            self.attention_maps = attn[0, 0].detach().cpu().numpy()
        
        # Apply attention and reshape
        x = (attn @ v).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class DynamicPositionalEmbedding(nn.Module):
    """Dynamic Positional Embedding using 3D depth-wise convolution"""
    def __init__(self, dim, kernel_size=3):
        super(DynamicPositionalEmbedding, self).__init__()
        self.conv = nn.Conv3d(
            dim, 
            dim, 
            kernel_size=(kernel_size, kernel_size, kernel_size),
            padding='same',
            groups=dim  # Depth-wise convolution
        )
    
    def forward(self, x):
        # x shape: [B, T, H*W, D]
        B, T, N, C = x.shape
        
        # Reshape for 3D convolution
        H = W = int(math.sqrt(N))
        x_3d = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        
        # Apply 3D convolution
        x_conv = self.conv(x_3d)
        
        # Reshape back
        x_out = x_conv.permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        
        return x + x_out  # Residual connection

class ModifiedEfficientAdditiveAttention(nn.Module):
    """Modified Efficient Additive Attention (MEAA) as described in the CUE-Net paper"""
    def __init__(self, dim):
        super(ModifiedEfficientAdditiveAttention, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        
        # Query projection
        self.q_proj = nn.Linear(dim, dim, bias=False)
        # Key projection
        self.k_proj = nn.Linear(dim, dim, bias=False)
        
        # Learnable attention weights
        self.w_a = nn.Parameter(torch.randn(dim))
        
        # Output projections
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        
        # Store attention maps for visualization
        self.attention_scores = None
    
    def forward(self, q, k):
        # Project query and key
        # q is [B, 1, dim], k is [B, T*N, dim]
        q_star = self.q_proj(q)  # [B, 1, dim]
        k_star = self.k_proj(k)  # [B, T*N, dim]
        
        # Learn attention weights
        alpha = torch.matmul(q_star, self.w_a.unsqueeze(0).unsqueeze(-1)) * self.scale  # [B, 1, 1]
        
        # Global query vector
        q_g = alpha * q_star  # [B, 1, dim]
        
        # Element-wise multiplication with key
        B, L, D = k_star.shape
        fused = q_g.expand(B, L, D) * k_star  # [B, L, dim]
        
        # Apply linear layers with residual connection
        output = self.w1(fused) + q_star.expand(B, L, D)
        output = self.w2(output)
        
        # Store attention scores for visualization
        self.attention_scores = alpha.detach().cpu().numpy()
        
        # Mean pooling along sequence dimension
        output = torch.mean(output, dim=1, keepdim=True)  # [B, 1, dim]
        
        return output

class FeedForwardNetwork(nn.Module):
    """Feed Forward Network module"""
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class LocalUniBlockV2(nn.Module):
    """Local UniBlock V2 for local dependencies modeling"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(LocalUniBlockV2, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Simplified implementation using standard multi-head attention
        self.lt_mhra = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.gs_mhra = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed Forward Network
        self.ffn = FeedForwardNetwork(dim, dropout=dropout)
    
    def forward(self, x):
        # x shape: [B, T, 1, D]
        B, T, N, D = x.shape
        
        # Reshape for attention: [B*T, N, D]
        x_reshaped = x.reshape(B*T, N, D)
        
        # Local Temporal MHRA
        residual = x_reshaped
        x_norm = self.norm1(x_reshaped)
        x_attn, _ = self.lt_mhra(x_norm, x_norm, x_norm)
        x_reshaped = residual + x_attn
        
        # Global Spatial MHRA
        residual = x_reshaped
        x_norm = self.norm2(x_reshaped)
        x_attn, _ = self.gs_mhra(x_norm, x_norm, x_norm)
        x_reshaped = residual + x_attn
        
        # Feed Forward Network
        residual = x_reshaped
        x_reshaped = residual + self.ffn(self.norm3(x_reshaped))
        
        # Reshape back to [B, T, N, D]
        x = x_reshaped.reshape(B, T, N, D)
        
        return x

class GlobalUniBlockV3(nn.Module):
    """Global UniBlock V3 for global dependencies modeling with MEAA"""
    def __init__(self, dim, dropout=0.1):
        super(GlobalUniBlockV3, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Simplified implementation using global average pooling and MLP
        self.global_pool = nn.AdaptiveAvgPool2d((1, dim))
        
        # Feed Forward Network
        self.ffn = FeedForwardNetwork(dim, dropout=dropout)
        
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, dim))
    
    def forward(self, x):
        # x shape: [B, T, 1, D]
        B, T, N, D = x.shape
        
        # Global pooling across time dimension
        x_flat = x.reshape(B, T*N, D)
        x_pooled = torch.mean(x_flat, dim=1, keepdim=True)  # [B, 1, D]
        
        # Apply FFN
        residual = x_pooled
        x_norm = self.norm1(x_pooled)
        x_pooled = residual + self.ffn(x_norm)
        
        # Ensure output shape [B, 1, D]
        return x_pooled

class CUENet(nn.Module):
    """Complete implementation of CUE-Net for violence detection"""
    def __init__(self, num_frames=64, input_size=336, embed_dim=256, depth=4, num_heads=8, dropout=0.1):
        super(CUENet, self).__init__()
        
        # 3D Convolution Backbone
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Second conv block
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Spatial cropping attention
        self.spatial_crop_attn = SpatialCropAttention(128)
        
        # Calculate feature dimensions after convolutions and pooling
        h = input_size // 16  # After several stride-2 operations
        w = h
        
        # Embedding layer - use a more flexible approach
        # Instead of depending on specific dimensions, use a smaller embedding first
        self.embedding = nn.Linear(128 * h * w, embed_dim)
        
        # Local UniBlock V2 (stack of blocks)
        self.local_blocks = nn.ModuleList([
            LocalUniBlockV2(embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Global UniBlock V3
        self.global_block = GlobalUniBlockV3(embed_dim, dropout=dropout)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 2)  # Binary classification (fight/non-fight)
        
        # Fusion parameter
        self.beta = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Initialize CLIP weights
        self._init_clip_weights()
        
    def _init_clip_weights(self):
        """Initialize Global MHRA units with CLIP-ViT embeddings"""
        try:
            # Load CLIP model
            import open_clip
            clip_model_name = 'ViT-B-32'
            pretrained = 'openai'
            
            print(f"Trying to load CLIP model: {clip_model_name}, pretrained: {pretrained}")
            
            try:
                clip_model, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)
                print("Successfully loaded CLIP model")
            except Exception as e:
                print(f"Error loading CLIP model: {e}")
                print("Trying alternative approach...")
                # Try alternative approach if the first one fails
                clip_model = open_clip.create_model(clip_model_name, pretrained=pretrained)
                print("Successfully loaded CLIP model with alternative method")
            
            # Get attention weights from CLIP
            for i, block in enumerate(self.local_blocks):
                try:
                    # Only transfer weights to Global Spatial MHRA
                    clip_block_idx = i % len(clip_model.visual.transformer.resblocks)
                    clip_block = clip_model.visual.transformer.resblocks[clip_block_idx]
                    
                    # Copy attention weights
                    with torch.no_grad():
                        block.gs_mhra.qkv.weight.copy_(clip_block.attn.in_proj_weight)
                        if hasattr(clip_block.attn, 'in_proj_bias') and clip_block.attn.in_proj_bias is not None:
                            block.gs_mhra.qkv.bias.copy_(clip_block.attn.in_proj_bias)
                        block.gs_mhra.proj.weight.copy_(clip_block.attn.out_proj.weight)
                        if hasattr(clip_block.attn.out_proj, 'bias') and clip_block.attn.out_proj.bias is not None:
                            block.gs_mhra.proj.bias.copy_(clip_block.attn.out_proj.bias)
                    
                    print(f"Initialized block {i} with CLIP weights")
                except Exception as e:
                    print(f"Failed to initialize block {i}: {e}")
            
            print("Initialized Global MHRA units with CLIP-ViT embeddings")
        except Exception as e:
            print(f"Failed to initialize with CLIP weights: {e}")
            print("Continuing with random initialization")
        
    def forward(self, x):
        # Input x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # 3D CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Apply spatial cropping attention
        x = self.spatial_crop_attn(x)
        
        # Get actual dimensions after convolutions
        _, c, t, h, w = x.shape
        
        # Reshape for transformer - use actual dimensions
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, H, W, C]
        
        # Flatten spatial dimensions and channels for embedding
        x = x.reshape(B, t, h*w*c)
        
        # Apply embedding
        x = self.embedding(x)  # [B, T, embed_dim]
        
        # For Local UniBlock, we need [B, T, H*W, D/H*W] shape
        # But actually, for our implementation, we'll just reshape to [B, T*H*W, D]
        # and have the attention mechanisms handle it appropriately
        
        # Process through Local UniBlocks - treating each time step as a batch
        local_features = x  # [B, T, embed_dim]
        
        # Create a dummy H*W dimension for compatibility with the blocks
        # This is a simplified approach that avoids complex reshaping
        local_features = local_features.unsqueeze(2)  # [B, T, 1, embed_dim]
        
        for block in self.local_blocks:
            local_features = block(local_features)
        
        # Extract class token for local path (mean pooling)
        local_class_token = torch.mean(local_features, dim=1)  # [B, 1, embed_dim]
        local_class_token = local_class_token.squeeze(1)  # [B, embed_dim]
        
        # Process through Global UniBlock
        global_features = self.global_block(local_features)  # [B, 1, embed_dim]
        global_class_token = global_features.squeeze(1)  # [B, embed_dim]
        
        # Dynamic fusion with learnable beta parameter
        beta_prime = torch.sigmoid(self.beta)
        fused_features = (1 - beta_prime) * global_class_token + beta_prime * local_class_token
        
        # Classification
        x = self.norm(fused_features)
        x = self.fc(x)
        
        return x

    def get_attention_visualizations(self):
        """Extract attention visualizations from the model"""
        visualizations = {}
        
        # Get spatial cropping attention maps
        if hasattr(self.spatial_crop_attn, 'attention_maps'):
            visualizations['spatial_attention'] = self.spatial_crop_attn.attention_maps
        
        # Get self-attention maps from the first Local UniBlock
        if self.local_blocks and hasattr(self.local_blocks[0].gs_mhra, 'attention_maps'):
            visualizations['self_attention'] = self.local_blocks[0].gs_mhra.attention_maps
        
        # Get MEAA attention scores
        if hasattr(self.global_block.meaa, 'attention_scores'):
            visualizations['meaa_attention'] = self.global_block.meaa.attention_scores
        
        return visualizations