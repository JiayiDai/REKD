import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import nns.cnn as cnn
import nns.bilstm as bilstm
import nns.bert as bert
import nns.resnet as resnet
import nns.vit as vit

class Encoder(nn.Module):
    def __init__(self, args, embeddings=None, if_t=False):
        super(Encoder, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=True)
            self.fc = nn.Linear( len(args.filters)*args.filter_num,  args.hidden_dim)
            self.hidden = nn.Linear(args.hidden_dim, args.num_class)
        elif args.model_form == "bilstm":
            self.bilstm = bilstm.BILSTM(args, bilstm_dim=args.bilstm_dim, encoding=True)
            self.fc = nn.Linear(args.bilstm_dim * 2, args.num_class)
        elif "bert" in args.model_form.lower():
            self.bert = bert.BERT(args, encoding=True, if_t=if_t)
            self.fc = nn.Linear(self.bert.embed_dim, args.num_class)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0.0)
            self.embedding_layer = self.bert.embedding
            self.embedding_layer.weight.requires_grad = True
        elif "resnet" in args.model_form or "efficientnet" in self.args.model_form.lower():
            self.resnet = resnet.RESNET(args, encoding=True, if_t=if_t)
        elif "vit" in args.model_form.lower():
            self.vit = vit.ViT(args, encoding=True, if_t=if_t)
            self.classifier = nn.Linear(self.vit.hidden_size, args.num_class)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.constant_(self.classifier.bias, 0.0)
            self.embedding_layer = self.vit.embedding
            for param in self.embedding_layer.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError("Model form {} not yet supported!".format(args.model_form))

    def forward(self, x_indx, att_mask=None, mask=None):
        '''
            x_indx:  batch of word indices or pixels
            mask: Mask to apply over embeddings for rationales
        '''
        if "bert" in self.args.model_form.lower():
            x = self.embedding_layer(x_indx.squeeze(1))
        elif "vit" in self.args.model_form.lower():
            if mask != None:
                x = self.embedding_layer.patch_embeddings(x_indx)#.squeeze(1)
            else:
                x = self.embedding_layer(x_indx)
        else:
            x = x_indx
        if self.args.cuda:
            x = x.cuda()
            if mask != None:
                mask = mask.cuda()
                
        if not mask is None:
            x = x * mask.unsqueeze(-1)
            if 'vit' in self.args.model_form.lower():
                cls_tokens = self.embedding_layer.cls_token.expand(x_indx.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.embedding_layer.position_embeddings
                x = self.embedding_layer.dropout(x)
        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.cnn(x)
            hidden = F.relu( self.fc(hidden) )
            hidden = self.dropout(hidden)
            logit = self.hidden(hidden)
        elif self.args.model_form == 'bilstm':
            hidden = self.bilstm(x)
            logit = self.fc(self.dropout(hidden))
        elif 'bert' in self.args.model_form.lower():
            hidden = self.bert(x_indx, att_mask, inputs_embeds=x)
            hidden = self.dropout(hidden)
            logit = self.fc(hidden)
        elif 'resnet' in self.args.model_form.lower() or "efficientnet" in self.args.model_form.lower():
            logit = self.resnet(x)
        elif 'vit' in self.args.model_form.lower():
            hidden = self.vit(x)
            hidden = self.dropout(hidden)
            logit = self.classifier(hidden)
        else:
            raise Exception("Model form {} not yet supported for encoder!".format(self.args.model_form))
        return logit
