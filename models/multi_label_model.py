from transformers import AutoModel
from torch import nn
import numpy as np

# Multitask BERT classification model with four heads
class MultiLabelModel(nn.Module):

    def __init__(self, args):

        super(MultiLabelModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_ckpt, return_dict=False)
        self.drop = nn.Dropout(args.dropout_prob)

        # Classification head
        self.fc = nn.Linear(self.bert.config.hidden_size, args.n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        dropoutput = self.drop(pooled_output)

        output = self.fc(dropoutput)


        return output

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)