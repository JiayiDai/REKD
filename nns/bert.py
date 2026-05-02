from torch import nn
from transformers import AutoModel

class BERT(nn.Module):
    def __init__(self, args, encoding=None, if_t=False):
        super(BERT, self).__init__()
        if if_t:
            self.bert = AutoModel.from_pretrained(args.model_form_t, cache_dir="hf_cache", local_files_only=True)
        else:
            self.bert = AutoModel.from_pretrained(args.model_form, cache_dir="hf_cache", local_files_only=True)
        self.embedding = self.bert.embeddings.word_embeddings
        self.embed_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.encoding = encoding
        
    def forward(self, input_ids, att_mask, inputs_embeds=None):
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=att_mask)#,
        if self.encoding:#enc
            return outputs.last_hidden_state[:, 0]
        else:#gen
            return outputs.last_hidden_state
