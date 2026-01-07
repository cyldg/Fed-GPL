import torch
import torch.nn as nn
from functools import reduce
import math
from operator import mul

import timm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class PGVIT(VisionTransformer):

    def __init__(self,  hidden_dims=768,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=3,
                 VPT_type="Shallow", basic_state_dict=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)
        prompt_dim = hidden_dims
        self.prompt_proj = nn.Identity()
        patch_size = (img_size, img_size)
        num_tokens = Prompt_Token_num
        self.num_tokens = num_tokens  # number of prompted tokens

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.agnostic_prompt_basis = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
        
        nn.init.uniform_(self.agnostic_prompt_basis.data, -val, val)

        
        # initiate prompt
        # self.num_clients = num_clients
        # self.client_descriptors = nn.Parameter(torch.zeros(1, num_clients, prompt_dim))
        # nn.init.uniform_(self.client_descriptors.data, -val, val)

        # initiate hed, w_q, w_k, w_v
        # TODO
        self.num_heads = 12
        self.attention_head_dims = int(hidden_dims / self.num_heads)
        self.all_head_dims = self.num_heads * self.attention_head_dims

        self.w_q = nn.Linear(196, 1)
        self.w_k = nn.Linear(prompt_dim, self.all_head_dims)
        self.w_v = nn.Linear(prompt_dim, self.all_head_dims)
        self.out = nn.Linear(hidden_dims, self.num_tokens*hidden_dims)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_dims)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def New_CLS_head(self, new_classes=10):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        # self.Prompt_Tokens.requires_grad = True
        self.agnostic_prompt_basis.requires_grad = True
        self.w_q.weight.requires_grad = True
        self.w_q.bias.requires_grad = True
        self.w_k.weight.requires_grad = True
        self.w_k.bias.requires_grad = True
        self.w_v.weight.requires_grad = True
        self.w_v.bias.requires_grad = True
        self.out.weight.requires_grad = True
        self.out.bias.requires_grad = True

        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'agnostic_prompt_basis': self.agnostic_prompt_basis}
        # print(prompt_state_dict)
        return prompt_state_dict
    
    def obtain_model(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'agnostic_prompt_basis': self.agnostic_prompt_basis,
                             'w_q':self.w_q.state_dict(),
                             'w_k':self.w_k.state_dict(),
                             'w_v':self.w_v.state_dict(),
                             'out':self.out.state_dict(),
                             }
        # print(prompt_state_dict)
        return prompt_state_dict
    def load_model(self,model_state_dict):
        try:
            self.head.load_state_dict(model_state_dict['head'], False)
            self.head.load_state_dict(model_state_dict['w_q'], False)
            self.head.load_state_dict(model_state_dict['w_k'], False)
            self.head.load_state_dict(model_state_dict['w_v'], False)
            self.head.load_state_dict(model_state_dict['out'], False)
        except:
            print('head not match, so skip head')
        # else:
            # print('prompt head match')

        if self.agnostic_prompt_basis.shape == model_state_dict['agnostic_prompt_basis'].shape:

            # device check
            agnostic_prompt_basis = nn.Parameter(model_state_dict['agnostic_prompt_basis'])
            agnostic_prompt_basis.to(torch.device(self.agnostic_prompt_basis.device))

            self.agnostic_prompt_basis = agnostic_prompt_basis

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.agnostic_prompt_basis.shape)
            print('shape of model given prompt', prompt_state_dict['agnostic_prompt_basis'].shape)
            print('')   

    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        # else:
            # print('prompt head match')

        if self.agnostic_prompt_basis.shape == prompt_state_dict['agnostic_prompt_basis'].shape:

            # device check
            agnostic_prompt_basis = nn.Parameter(prompt_state_dict['agnostic_prompt_basis'])
            agnostic_prompt_basis.to(torch.device(self.agnostic_prompt_basis.device))

            self.agnostic_prompt_basis = agnostic_prompt_basis

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.agnostic_prompt_basis.shape)
            print('shape of model given prompt', prompt_state_dict['agnostic_prompt_basis'].shape)
            print('')

    def forward_features(self, x,Prompt_Tokens):
        # x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)


        Prompt_Token_num = Prompt_Tokens.shape[1]
        #print('Prompt_Token_num:',Prompt_Token_num)

        # concatenate Prompt_Tokens
        Prompt_Tokens = Prompt_Tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((x, Prompt_Tokens), dim=1)
        num_tokens = x.shape[1]
        # Sequntially procees
        x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x
    
    def forward(self,x):
        # b num_clients num_tokens d*h
        xx = self.patch_embed(x) # 196*768
        # print("x:",x.size())
        x = xx.permute(0, 2, 1)

        mixed_q_layer = self.w_q(x) 
        mixed_q_layer = mixed_q_layer.permute(0, 2, 1)
        # print(mixed_q_layer.size())
        # mixed_q_layer = self.w_q(torch.reshape(x, (-1,1，196*768))) 
        mixed_k_layer = self.w_k(self.agnostic_prompt_basis) # 1, num_tokens, num_heads * attention_head_dims
        mixed_v_layer = self.w_v(self.agnostic_prompt_basis)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        attention_scores = torch.matmul(q_layer,k_layer.transpose(-1, -2)) # 1, num_head, num_clients, num_tokens
        attention_scores = attention_scores / math.sqrt(self.attention_head_dims)

        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs,v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #print(context_layer.size())
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dims,)
        #print(new_context_layer_shape,context_layer.size()[:-2] , (self.all_head_dims,))
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output =  attention_output.reshape(-1, self.num_tokens, 768)


        specific_prompts = self.agnostic_prompt_basis + attention_output
        # print(specific_prompts.size())
        x = self.forward_features(xx,specific_prompts)

        x = self.fc_norm(x[:, 0, :])  # fixme for old timm: x = self.pre_logits(x[:, 0, :])
        x = self.head(x)
        return x




class PGVITBdeep(VisionTransformer):

    def __init__(self,  hidden_dims=768,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=3,
                 VPT_type="Shallow", basic_state_dict=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)
        prompt_dim = hidden_dims
        self.prompt_proj = nn.Identity()
        patch_size = (img_size, img_size)
        num_tokens = Prompt_Token_num
        self.num_tokens = num_tokens  # number of prompted tokens

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.agnostic_prompt_basis = nn.Parameter(torch.zeros(1, len(self.blocks)*num_tokens, prompt_dim))
        #print(self.agnostic_prompt_basis.size())
        
        nn.init.uniform_(self.agnostic_prompt_basis.data, -val, val)

        
        # initiate prompt
        # self.num_clients = num_clients
        # self.client_descriptors = nn.Parameter(torch.zeros(1, num_clients, prompt_dim))
        # nn.init.uniform_(self.client_descriptors.data, -val, val)

        # initiate hed, w_q, w_k, w_v
        # TODO
        self.num_heads = 12
        self.attention_head_dims = int(hidden_dims / self.num_heads)
        self.all_head_dims = self.num_heads * self.attention_head_dims

        self.w_q = nn.Linear(196, 1)
        self.w_k = nn.Linear(prompt_dim, self.all_head_dims)
        self.w_v = nn.Linear(prompt_dim, self.all_head_dims)
        self.out = nn.Linear(hidden_dims, len(self.blocks)*self.num_tokens*hidden_dims)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_dims)
        x = x.view(*new_x_shape)
        # print(x.shape)
        return x.permute(0, 2, 1, 3)

    def New_CLS_head(self, new_classes=10):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        # self.Prompt_Tokens.requires_grad = True
        self.agnostic_prompt_basis.requires_grad = True
        self.w_q.weight.requires_grad = True
        self.w_q.bias.requires_grad = True
        self.w_k.weight.requires_grad = True
        self.w_k.bias.requires_grad = True
        self.w_v.weight.requires_grad = True
        self.w_v.bias.requires_grad = True
        self.out.weight.requires_grad = True
        self.out.bias.requires_grad = True

        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'agnostic_prompt_basis': self.agnostic_prompt_basis}
        # print(prompt_state_dict)
        return prompt_state_dict
    
    def obtain_model(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'agnostic_prompt_basis': self.agnostic_prompt_basis,
                             'w_q':self.w_q.state_dict(),
                             'w_k':self.w_k.state_dict(),
                             'w_v':self.w_v.state_dict(),
                             'out':self.out.state_dict(),
                             }
        # print(prompt_state_dict)
        return prompt_state_dict
    def load_model(self,model_state_dict):
        try:
            self.head.load_state_dict(model_state_dict['head'], False)
            self.head.load_state_dict(model_state_dict['w_q'], False)
            self.head.load_state_dict(model_state_dict['w_k'], False)
            self.head.load_state_dict(model_state_dict['w_v'], False)
            self.head.load_state_dict(model_state_dict['out'], False)
        except:
            print('head not match, so skip head')
        # else:
            # print('prompt head match')

        if self.agnostic_prompt_basis.shape == model_state_dict['agnostic_prompt_basis'].shape:

            # device check
            agnostic_prompt_basis = nn.Parameter(model_state_dict['agnostic_prompt_basis'])
            agnostic_prompt_basis.to(torch.device(self.agnostic_prompt_basis.device))

            self.agnostic_prompt_basis = agnostic_prompt_basis

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.agnostic_prompt_basis.shape)
            print('shape of model given prompt', prompt_state_dict['agnostic_prompt_basis'].shape)
            print('')   

    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        # else:
            # print('prompt head match')

        if self.agnostic_prompt_basis.shape == prompt_state_dict['agnostic_prompt_basis'].shape:

            # device check
            agnostic_prompt_basis = nn.Parameter(prompt_state_dict['agnostic_prompt_basis'])
            agnostic_prompt_basis.to(torch.device(self.agnostic_prompt_basis.device))

            self.agnostic_prompt_basis = agnostic_prompt_basis

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.agnostic_prompt_basis.shape)
            print('shape of model given prompt', prompt_state_dict['agnostic_prompt_basis'].shape)
            print('')

    def forward_features(self, x,Prompt_Tokens):
        # x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        Prompt_Token_num = Prompt_Tokens.shape[2]
        #print('Prompt_Token_num',Prompt_Token_num)
        Prompt_Tokens=list(torch.chunk(Prompt_Tokens, chunks=len(self.blocks), dim=1))
        Prompt_Tokens = [t.squeeze(dim=1) for t in Prompt_Tokens]
        #print(len(Prompt_Tokens))
        #print(Prompt_Tokens[0].size())

        for i in range(len(self.blocks)):
            # concatenate Prompt_Tokens
            Prompt_Token = Prompt_Tokens[i]
            #print(x.shape,Prompt_Token.shape)
            # firstly concatenate
            x = torch.cat((x, Prompt_Token.expand(x.shape[0], -1, -1)), dim=1)
            num_tokens = x.shape[1]
            # lastly remove, a genius trick
            x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x
    
    def forward(self,x):
        # b num_clients num_tokens d*h
        x = self.patch_embed(x) # 196*768
        # print("x:",x.size())
        # x = xx.permute(0, 2, 1)

        # mixed_q_layer = self.w_q(x) 
        # mixed_q_layer = mixed_q_layer.permute(0, 2, 1)
        # print(mixed_q_layer.size())
        mixed_q_layer = self.w_q(torch.reshape(x, (-1,1,196*768))) 
        mixed_k_layer = self.w_k(self.agnostic_prompt_basis) # 1, num_tokens, num_heads * attention_head_dims
        mixed_v_layer = self.w_v(self.agnostic_prompt_basis)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)
        # print(mixed_q_layer.size(),mixed_k_layer.size(),mixed_v_layer.size())
        attention_scores = torch.matmul(q_layer,k_layer.transpose(-1, -2)) # 1, num_head, num_clients, num_tokens
        attention_scores = attention_scores / math.sqrt(self.attention_head_dims)

        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs,v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #print(context_layer.size())
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dims,)
        #print(new_context_layer_shape,context_layer.size()[:-2] , (self.all_head_dims,))
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        #print('attention',attention_output.size())
        attention_output =  attention_output.reshape(-1, len(self.blocks)*self.num_tokens, 768)
        #print('attention',attention_output.size())


        specific_prompts = self.agnostic_prompt_basis + attention_output
        specific_prompts = specific_prompts.reshape(-1,len(self.blocks),self.num_tokens,768)
        #print('specific',specific_prompts.size())
        x = self.forward_features(x,specific_prompts)


        x = self.fc_norm(x[:, 0, :])  # fixme for old timm: x = self.pre_logits(x[:, 0, :])
        x = self.head(x)
        return x




    def __init__(self,  hidden_dims=768,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=3,
                 VPT_type="Shallow", basic_state_dict=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)
        prompt_dim = hidden_dims
        self.prompt_proj = nn.Identity()
        patch_size = (img_size, img_size)
        num_tokens = Prompt_Token_num
        self.num_tokens = num_tokens  # number of prompted tokens

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.agnostic_prompt_basis = nn.Parameter(torch.zeros(1, len(self.blocks)*num_tokens, prompt_dim))
        #print(self.agnostic_prompt_basis.size())
        
        nn.init.uniform_(self.agnostic_prompt_basis.data, -val, val)

        
        # initiate prompt
        # self.num_clients = num_clients
        # self.client_descriptors = nn.Parameter(torch.zeros(1, num_clients, prompt_dim))
        # nn.init.uniform_(self.client_descriptors.data, -val, val)

        # initiate hed, w_q, w_k, w_v
        # TODO
        self.num_heads = 12
        self.attention_head_dims = int(hidden_dims / self.num_heads)
        self.all_head_dims = self.num_heads * self.attention_head_dims

        self.w_q = nn.Linear(prompt_dim*196, self.all_head_dims)
        # self.w_q = nn.Linear(196, 1)
        self.w_k = nn.Linear(prompt_dim, self.all_head_dims)
        self.w_v = nn.Linear(prompt_dim, self.all_head_dims)
        self.out = nn.Linear(hidden_dims, len(self.blocks)*self.num_tokens*hidden_dims)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_dims)
        x = x.view(*new_x_shape)
        # print(x.shape)
        return x.permute(0, 2, 1, 3)

    def New_CLS_head(self, new_classes=10):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        # self.Prompt_Tokens.requires_grad = True
        self.agnostic_prompt_basis.requires_grad = True
        self.w_q.weight.requires_grad = True
        self.w_q.bias.requires_grad = True
        self.w_k.weight.requires_grad = True
        self.w_k.bias.requires_grad = True
        self.w_v.weight.requires_grad = True
        self.w_v.bias.requires_grad = True
        self.out.weight.requires_grad = True
        self.out.bias.requires_grad = True

        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'agnostic_prompt_basis': self.agnostic_prompt_basis}
        # print(prompt_state_dict)
        return prompt_state_dict
    
    def obtain_model(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'agnostic_prompt_basis': self.agnostic_prompt_basis,
                             'w_q':self.w_q.state_dict(),
                             'w_k':self.w_k.state_dict(),
                             'w_v':self.w_v.state_dict(),
                             'out':self.out.state_dict(),
                             }
        # print(prompt_state_dict)
        return prompt_state_dict
    def load_model(self,model_state_dict):
        try:
            self.head.load_state_dict(model_state_dict['head'], False)
            self.head.load_state_dict(model_state_dict['w_q'], False)
            self.head.load_state_dict(model_state_dict['w_k'], False)
            self.head.load_state_dict(model_state_dict['w_v'], False)
            self.head.load_state_dict(model_state_dict['out'], False)
        except:
            print('head not match, so skip head')
        # else:
            # print('prompt head match')

        if self.agnostic_prompt_basis.shape == model_state_dict['agnostic_prompt_basis'].shape:

            # device check
            agnostic_prompt_basis = nn.Parameter(model_state_dict['agnostic_prompt_basis'])
            agnostic_prompt_basis.to(torch.device(self.agnostic_prompt_basis.device))

            self.agnostic_prompt_basis = agnostic_prompt_basis

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.agnostic_prompt_basis.shape)
            print('shape of model given prompt', prompt_state_dict['agnostic_prompt_basis'].shape)
            print('')   

    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        # else:
            # print('prompt head match')

        if self.agnostic_prompt_basis.shape == prompt_state_dict['agnostic_prompt_basis'].shape:

            # device check
            agnostic_prompt_basis = nn.Parameter(prompt_state_dict['agnostic_prompt_basis'])
            agnostic_prompt_basis.to(torch.device(self.agnostic_prompt_basis.device))

            self.agnostic_prompt_basis = agnostic_prompt_basis

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.agnostic_prompt_basis.shape)
            print('shape of model given prompt', prompt_state_dict['agnostic_prompt_basis'].shape)
            print('')

    def forward_features(self, x,Prompt_Tokens):
        # x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        Prompt_Token_num = Prompt_Tokens.shape[2]
        #print('Prompt_Token_num',Prompt_Token_num)
        Prompt_Tokens=list(torch.chunk(Prompt_Tokens, chunks=len(self.blocks), dim=1))
        Prompt_Tokens = [t.squeeze(dim=1) for t in Prompt_Tokens]
        #print(len(Prompt_Tokens))
        #print(Prompt_Tokens[0].size())

        for i in range(len(self.blocks)):
            # concatenate Prompt_Tokens
            Prompt_Token = Prompt_Tokens[i]
            #print(x.shape,Prompt_Token.shape)
            # firstly concatenate
            x = torch.cat((x, Prompt_Token.expand(x.shape[0], -1, -1)), dim=1)
            num_tokens = x.shape[1]
            # lastly remove, a genius trick
            x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]
        # Prompt_Token_num = Prompt_Tokens.shape[2]
        # print('Prompt_Token_num',Prompt_Token_num)
        # Prompt_Tokens=torch.chunk(Prompt_Tokens, chunks=len(self.blocks), dim=1)
        # print(len(Prompt_Tokens))
        # print(Prompt_Tokens[0].size())

        # for i in range(len(self.blocks)):
        #     # concatenate Prompt_Tokens
        #     Prompt_Token = Prompt_Tokens[i].unsqueeze(0)
        #     print(x.shape,Prompt_Token.shape)
        #     # firstly concatenate
        #     x = torch.cat((x, Prompt_Token.expand(x.shape[0], -1, -1)), dim=1)
        #     num_tokens = x.shape[1]
        #     # lastly remove, a genius trick
        #     x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]
        x = self.norm(x)
        return x
    
    def forward(self,x):
        # b num_clients num_tokens d*h
        x = self.patch_embed(x) # 196*768
        # print("x:",x.size())
        # x = xx.permute(0, 2, 1)

        # mixed_q_layer = self.w_q(x) 
        # mixed_q_layer = mixed_q_layer.permute(0, 2, 1)
        # print(mixed_q_layer.size())
        mixed_q_layer = self.w_q(torch.reshape(x, (-1,1,196*768))) 
        mixed_k_layer = self.w_k(self.agnostic_prompt_basis) # 1, num_tokens, num_heads * attention_head_dims
        mixed_v_layer = self.w_v(self.agnostic_prompt_basis)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)
        # print(mixed_q_layer.size(),mixed_k_layer.size(),mixed_v_layer.size())
        attention_scores = torch.matmul(q_layer,k_layer.transpose(-1, -2)) # 1, num_head, num_clients, num_tokens
        attention_scores = attention_scores / math.sqrt(self.attention_head_dims)

        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs,v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #print(context_layer.size())
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dims,)
        #print(new_context_layer_shape,context_layer.size()[:-2] , (self.all_head_dims,))
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        #print('attention',attention_output.size())
        attention_output =  attention_output.reshape(-1, len(self.blocks)*self.num_tokens, 768)
        #print('attention',attention_output.size())


        specific_prompts = self.agnostic_prompt_basis + attention_output
        specific_prompts = specific_prompts.reshape(-1,len(self.blocks),self.num_tokens,768)
        #print('specific',specific_prompts.size())
        x = self.forward_features(x,specific_prompts)


        x = self.fc_norm(x[:, 0, :])  # fixme for old timm: x = self.pre_logits(x[:, 0, :])
        return x
if __name__ == '__main__':
    num_classes=10
    edge_size=224
    VPT_type="deep"
    patch_size = 16
    Prompt_Token_num=2
    basic_model = timm.create_model('vit_base_patch' + str(patch_size) + '_' + str(edge_size),
                                    pretrained=True, pretrained_cfg_overlay = dict(file="/home/yuliang_chen/FedVPT/pytorch_model.bin"))

    model = PGVIT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                    VPT_type=VPT_type)

    model.load_state_dict(basic_model.state_dict(), False)
    model.New_CLS_head(num_classes)
    model.Freeze()


    img = torch.randn(1, 3, edge_size, edge_size)
    preds = model(img)  # (1, class_number)
    print('test model output:', preds)

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(param.numel())
            print()

    for name, param in basic_model.named_parameters():
        if 'block' not in name:
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {param_count}")

    param_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
    print(f"Parameter sizes: {param_sizes}")
    dict = model.obtain_model()
    print(dict.keys())

