import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from torch.nn import functional as F


class Self_Attention(nn.Module):
    def __init__(self, emb_size, head, dropout):
        super(Self_Attention, self).__init__()

        self.num_attention_heads = head
        self.attention_head_size = emb_size // head
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(dropout)

        # Add normalization layers
        self.norm = nn.LayerNorm(emb_size)
        self.input_norm = nn.LayerNorm(emb_size)

        self.query = nn.Sequential(
            nn.Linear(emb_size, self.all_head_size),
            nn.LayerNorm(self.all_head_size)
        )
        self.key = nn.Sequential(
            nn.Linear(emb_size, self.all_head_size),
            nn.LayerNorm(self.all_head_size)
        )
        self.value = nn.Sequential(
            nn.Linear(emb_size, self.all_head_size),
            nn.LayerNorm(self.all_head_size)
        )

        self.dense1 = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LayerNorm(emb_size),
            nn.ReLU()
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input):
        # Input normalization
        input = self.input_norm(input)
        
        # Project queries, keys, and values
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input)

        # Transpose for attention computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores with gradient clipping
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = torch.clamp(attention_scores, -10.0, 10.0)  # Prevent extreme values
        
        # Apply softmax and dropout
        attention_probs = self.dropout(nn.Softmax(dim=-1)(attention_scores))
        
        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply final transformations
        out = context_layer + self.dense1(context_layer)
        out = self.norm(out)
        
        return out
    
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_t, self.d_a, self.d_v, self.d_target = args.dim_t, args.dim_a, args.dim_v, args.emb_size
        self.att_dp = args.atten_dropout
        self.n_head = args.n_head
        
        # Initialize with smaller values and better normalization
        self.text_map = nn.Sequential(
            nn.Linear(self.d_t, self.d_target),
            nn.LayerNorm(self.d_target),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.audio_map = nn.Sequential(
            nn.Linear(self.d_a, self.d_target),
            nn.LayerNorm(self.d_target),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.vision_map = nn.Sequential(
            nn.Linear(self.d_v, self.d_target),
            nn.LayerNorm(self.d_target),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Initialize attention with smaller values
        self.attention_text = Self_Attention(self.d_target, self.n_head, self.att_dp)
        self.attention_audio = Self_Attention(self.d_target, self.n_head, self.att_dp)
        self.attention_vision = Self_Attention(self.d_target, self.n_head, self.att_dp)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, text, audio, vision):
        # Input validation and normalization
        if torch.isnan(text).any() or torch.isnan(audio).any() or torch.isnan(vision).any():
            print("Warning: NaN values in input tensors")
            text = torch.nan_to_num(text, 0.0)
            audio = torch.nan_to_num(audio, 0.0)
            vision = torch.nan_to_num(vision, 0.0)
        
        # Normalize inputs
        text = F.normalize(text, p=2, dim=-1)
        audio = F.normalize(audio, p=2, dim=-1)
        vision = F.normalize(vision, p=2, dim=-1)
        
        # Map to target dimension with gradient clipping
        text = torch.clamp(self.text_map(text), -1.0, 1.0)
        audio = torch.clamp(self.audio_map(audio), -1.0, 1.0)
        vision = torch.clamp(self.vision_map(vision), -1.0, 1.0)
        
        # Add sequence dimension if needed
        if len(text.shape) == 2:
            text = text.unsqueeze(1)
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(1)
        if len(vision.shape) == 2:
            vision = vision.unsqueeze(1)
            
        # Apply attention with gradient clipping
        shared_text = torch.clamp(self.attention_text(text), -1.0, 1.0)
        shared_audio = torch.clamp(self.attention_audio(audio), -1.0, 1.0)
        shared_vision = torch.clamp(self.attention_vision(vision), -1.0, 1.0)
        
        # Compute different representations
        diff_text = text - shared_text
        diff_audio = audio - shared_audio
        diff_vision = vision - shared_vision
        
        # Normalize representations
        shared_text = F.normalize(shared_text, p=2, dim=-1)
        shared_audio = F.normalize(shared_audio, p=2, dim=-1)
        shared_vision = F.normalize(shared_vision, p=2, dim=-1)
        diff_text = F.normalize(diff_text, p=2, dim=-1)
        diff_audio = F.normalize(diff_audio, p=2, dim=-1)
        diff_vision = F.normalize(diff_vision, p=2, dim=-1)
        
        # Concatenate representations
        shared_emb = torch.cat((shared_text, shared_audio, shared_vision), 1)
        diff_emb = torch.cat((diff_text, diff_audio, diff_vision), 1)
        
        # Final validation
        if torch.isnan(shared_emb).any() or torch.isnan(diff_emb).any():
            print("Warning: NaN values in output tensors")
            shared_emb = torch.nan_to_num(shared_emb, 0.0)
            diff_emb = torch.nan_to_num(diff_emb, 0.0)
            
        return shared_emb, diff_emb

class Actor(nn.Module):
    def __init__(self, args):
        super().__init__()
        d_target = args.emb_size
        self.length = args.seqlength
        num_layer = 1
        att_dp = args.atten_dropout
        dp = args.dropout
        n_head = args.n_head

        # Add normalization layers
        self.state_norm = nn.LayerNorm(d_target)

        ####generate action for different representations
        self.text_action = nn.Sequential(
            nn.Linear(d_target, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Linear(d_target, 1),
            nn.Tanh()  # Bound actions between -1 and 1
        )
        self.audio_action = nn.Sequential(
            nn.Linear(d_target, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Linear(d_target, 1),
            nn.Tanh()
        )
        self.vision_action = nn.Sequential(
            nn.Linear(d_target, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Linear(d_target, 1),
            nn.Tanh()
        )

        layer = nn.TransformerEncoderLayer(d_model=d_target, nhead=n_head, dropout=att_dp, batch_first=True)
        self.temporal_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.temporal_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.temporal_v = nn.TransformerEncoder(layer, num_layers=num_layer)

        self.weight_t = nn.Sequential(
            nn.Linear(d_target * 2, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Linear(d_target, 1),
            nn.Sigmoid()
        )
        self.weight_a = nn.Sequential(
            nn.Linear(d_target * 2, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Linear(d_target, 1),
            nn.Sigmoid()
        )
        self.weight_v = nn.Sequential(
            nn.Linear(d_target * 2, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Linear(d_target, 1),
            nn.Sigmoid()
        )

        self.predict = nn.Sequential(
            nn.Linear(d_target * 2, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(d_target, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(d_target, 2)
        )

    def forward(self, state):
        # Normalize input state
        state = self.state_norm(state)
        
        state_t, state_a, state_v = state[:, :self.length], state[:, self.length:self.length * 2], state[:, self.length*2:]
        action_t = self.text_action(state_t)
        action_a = self.audio_action(state_a)
        action_v = self.vision_action(state_v)
        action = torch.cat((action_t, action_a, action_v), 1)
        return action

    def update_state(self, state, action):
        state_t, state_a, state_v = state[:, :self.length], state[:, self.length:self.length * 2], state[:, self.length*2:]
        action_t, action_a, action_v = action[:, :self.length], action[:, self.length:self.length * 2], action[:, self.length*2:]
        
        # Clip actions to prevent extreme values
        action_t = torch.clamp(action_t, -1.0, 1.0)
        action_a = torch.clamp(action_a, -1.0, 1.0)
        action_v = torch.clamp(action_v, -1.0, 1.0)
        
        diff_t = torch.mul(state_t, action_t)
        diff_a = torch.mul(state_a, action_a)
        diff_v = torch.mul(state_v, action_v)
        state = torch.cat((diff_t, diff_a, diff_v), 1)
        return state
    
    def predictor(self, shared_embs, diff_embs):
        self.batch = shared_embs.shape[0]
        shared_t, shared_a, shared_v = shared_embs[:, :self.length], shared_embs[:, self.length:self.length * 2], shared_embs[:, self.length*2:]
        diff_t, diff_a, diff_v = diff_embs[:, :self.length], diff_embs[:, self.length:self.length * 2], diff_embs[:, self.length*2:]

        shared_t = shared_t.mean(1)
        shared_a = shared_a.mean(1)
        shared_v = shared_v.mean(1)

        diff_t = self.temporal_t(diff_t)
        diff_a = self.temporal_a(diff_a)
        diff_v = self.temporal_v(diff_v)
        
        diff_t = diff_t.mean(1)
        diff_a = diff_a.mean(1)
        diff_v = diff_v.mean(1)

        emb_t = torch.cat((shared_t, diff_t), -1)
        emb_a = torch.cat((shared_a, diff_a), -1)
        emb_v = torch.cat((shared_v, diff_v), -1)

        weight_t = self.weight_t(emb_t)
        weight_a = self.weight_a(emb_a)
        weight_v = self.weight_v(emb_v)
        weights = torch.cat((weight_t, weight_a, weight_v), -1)
        
        emb_t = torch.mul(emb_t, weights[:, 0].unsqueeze(1))
        emb_a = torch.mul(emb_a, weights[:, 1].unsqueeze(1))
        emb_v = torch.mul(emb_v, weights[:, 2].unsqueeze(1))
        
        predictions = self.predict((emb_t + emb_a + emb_v)/3)
        return predictions

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        d_target = args.emb_size
        self.length = args.seqlength
        dp = args.dropout
        att_dp = args.atten_dropout
        self.n_head = args.n_head 
        num_layer = 1

        # Add normalization layers
        self.state_norm = nn.LayerNorm(d_target)
        self.action_norm = nn.LayerNorm(d_target)

        layer = nn.TransformerEncoderLayer(d_model=d_target + self.n_head, nhead=self.n_head, dropout=att_dp, batch_first=True)
        self.critic_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.critic_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.critic_v = nn.TransformerEncoder(layer, num_layers=num_layer)

        self.proj = nn.Sequential(
            nn.Linear((d_target + self.n_head) * 3, d_target * 2),
            nn.LayerNorm(d_target * 2),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(d_target * 2, d_target),
            nn.LayerNorm(d_target),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(d_target, 1))

    def forward(self, state, action):
        b = state.shape[0]
        
        # Normalize inputs
        state = self.state_norm(state)
        action = self.action_norm(action)
        
        state_t, state_a, state_v = state[:, :self.length], state[:, self.length:self.length * 2], state[:, self.length * 2:]
        action_t, action_a, action_v = action[:, :self.length], action[:, self.length:self.length * 2], action[:, self.length * 2:]
        
        # Clip actions to prevent extreme values
        action_t = torch.clamp(action_t, -1.0, 1.0)
        action_a = torch.clamp(action_a, -1.0, 1.0)
        action_v = torch.clamp(action_v, -1.0, 1.0)
        
        action_t = action_t.repeat(1, 1, self.n_head)
        action_a = action_a.repeat(1, 1, self.n_head)
        action_v = action_v.repeat(1, 1, self.n_head)
        
        text = torch.cat((state_t, action_t), -1)
        acoustic = torch.cat((state_a, action_a), -1)
        visual = torch.cat((state_v, action_v), -1)

        text = self.critic_t(text)
        visual = self.critic_v(visual)
        acoustic = self.critic_a(acoustic)

        text = text.mean(1)
        acoustic = acoustic.mean(1)
        visual = visual.mean(1)

        q = self.proj(torch.cat((text, acoustic, visual), -1))
        return torch.tanh(q.view(b, 1))  # Bound Q-values between -1 and 1