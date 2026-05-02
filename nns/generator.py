import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import nns.cnn as cnn
import nns.bilstm as bilstm
import nns.bert as bert
import nns.resnet as resnet
import nns.vit as vit

class Generator(nn.Module):
    def __init__(self, args, if_t=False):
        super(Generator, self).__init__()
        self.args = args
        self.z_dim = 2#bernoulli distri for feature selection

        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time = False)
            self.fc = nn.Linear((len(args.filters)* args.filter_num), self.z_dim)
        elif args.model_form == 'bilstm':
            self.bilstm = bilstm.BILSTM(args)
            self.fc = nn.Linear(args.bilstm_dim*2, self.z_dim)
        elif "bert" in args.model_form.lower():
            self.bert = bert.BERT(args, encoding=False, if_t=if_t)
            self.fc = nn.Linear(256, self.z_dim)
            self.feature_proj = nn.Linear(self.bert.embed_dim, 256)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0.0)
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.constant_(self.feature_proj.bias, 0.0)
            self.embedding_layer = self.bert.embedding
            self.embedding_layer.weight.requires_grad = False
        elif "resnet" in args.model_form.lower() or "efficientnet" in self.args.model_form.lower():
            self.resnet = resnet.RESNET(args, encoding=False, if_t=if_t)
        elif "vit" in self.args.model_form.lower():
            self.vit = vit.ViT(args, encoding=False, if_t=if_t)
            self.fc = nn.Linear(self.vit.hidden_size, self.z_dim)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0.0)
        else:
            raise NotImplementedError("Model form {} not yet supported!".format(args.model_form))


        self.dropout = nn.Dropout(args.dropout)

    def sample_gumbel(self, shape, eps=1e-20):
        #Samples from a Gumbel(0, 1) distribution.
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, dim=-1):
        """Draws a sample from the Gumbel-Softmax distribution."""
        gumbel_noise = self.sample_gumbel(logits.size())
        if logits.is_cuda:
            gumbel_noise = gumbel_noise.cuda()
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=dim), F.log_softmax(y / temperature, dim=dim)

    def gumbel_softmax(self, logits, temperature, hard=False, dim=-1):
        """
        Samples from the Gumbel-Softmax distribution and optionally discretizes.
        """
        y, log_y = self.gumbel_softmax_sample(logits, temperature, dim=dim)
        if not hard:
            return y, log_y
        
        shape = y.size()
        
        # Find the max along the specified dimension
        _, ind = y.max(dim=dim, keepdim=True)
        
        # Create one-hot encoding
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(dim, ind, 1.0)
        
        # Straight-through trick
        y_hard = (y_hard - y).detach() + y
        
        return y_hard, log_y

    def __z_forward(self, activ, model_form):
        '''
            Returns prob of each token being selected
        '''
        if model_form == "cnn":
            activ = activ.transpose(1,2)#batch*length*300
            dim = -1
        elif "bert" in self.args.model_form.lower():
            activ = self.feature_proj(activ)#batch*256*768 -> batch*256*256
            activ = self.dropout(F.relu(activ))
            logits = self.fc(activ)#batch*length*2
            dim = -1
        elif "resnet" in self.args.model_form.lower() or "efficientnet" in self.args.model_form.lower():
            logits = activ
            dim = 1
        elif "vit" in self.args.model_form.lower():
            activ = self.dropout(activ)
            logits = self.fc(activ)#batch*length*2
            dim = -1
        else:
            pass
        probs, log_probs = self.gumbel_softmax(logits, self.args.gumbel_t, hard=True, dim=dim)
        if "resnet" in self.args.model_form.lower() or "efficientnet" in self.args.model_form.lower():
            z = probs[:,1,:,:]#prob shape bz*2*M*M
        else:
            z = probs[:,:,1]#prob shape bz*L*2
        return z, probs, log_probs
    
    def forward(self, x_indx, att_mask=None):
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on x
        '''
        if "bert" in self.args.model_form.lower():
            x = self.embedding_layer(x_indx.squeeze(1))
        else:
            x = x_indx
        if self.args.cuda:
            x = x.cuda()
        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            activ = self.cnn(x)
            z, probs, log_probs = self.__z_forward(F.relu(activ), "cnn") 
        elif self.args.model_form == "bilstm":
            activ = self.bilstm(x)
            z, probs, log_probs = self.__z_forward(F.relu(activ), "bilstm")
        elif "bert" in self.args.model_form.lower():
            activ = self.bert(x_indx, att_mask, inputs_embeds=x)
            z, probs, log_probs = self.__z_forward(activ, "bert")
        elif "resnet" in self.args.model_form.lower() or "efficientnet" in self.args.model_form.lower():
            activ = self.resnet(x)#b*2*32*32
            z, probs, log_probs = self.__z_forward(activ, "resnet")
        elif "vit" in self.args.model_form.lower():
            activ = self.vit(x)
            z, probs, log_probs = self.__z_forward(activ, "vit")
        else:
            raise NotImplementedError("Model form {} not yet supported for generator!".format(self.args.model_form))
        mask = self.sample(z)
        return mask, probs, log_probs

    def get_hard_mask(self, z):
        with torch.no_grad():
            masked = torch.ge(z, 0.5).float()
        del z
        return masked

    def sample(self, z):
        '''
            Get mask from probablites at each token. Use gumbel
            softmax at train time, hard mask at test time
        '''
        mask = z
        if self.training:
            mask = z
        else:
            ## pointwise set <.5 to 0 >=.5 to 1
            mask = self.get_hard_mask(z)
        return mask

    def loss(self, mask, target_sparsity=0.1):
        '''
            Compute the generator specific costs
        '''
        if "resnet" in self.args.model_form.lower() or "efficientnet" in self.args.model_form.lower():
            selection = torch.mean(torch.sum(mask, dim=(1,2)))
        else:
            selection = torch.mean(torch.sum(mask, dim=1))
        selection_cost = (selection - self.args.target_sparsity*self.args.total_features) ** 2
        return selection, selection_cost
