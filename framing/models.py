import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaModel
from transformers import XGLMModel
from transformers import AutoConfig, AutoModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from config import PATH_TRANSFORMER_CACHE


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class FrameMultiClassification(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.modelname = config._name_or_path
        # self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        # self.roberta = AutoModel(config, add_pooling_layer=False)

        dropout_prob = config.dropout if "xglm" in config._name_or_path else config.hidden_dropout_prob
        self.classifier = ClassificationHead(config.hidden_size, dropout_prob, config.num_labels)

        if "xglm" in config._name_or_path:
            self.roberta = AutoModel.from_pretrained(config._name_or_path, config=config, cache_dir=PATH_TRANSFORMER_CACHE)
        else:
            self.roberta = AutoModel.from_pretrained(config._name_or_path, config=config, add_pooling_layer=False, cache_dir=PATH_TRANSFORMER_CACHE)

        self.initialize_classifier()
    
    def initialize_classifier(self):
        self._init_weights(self.classifier.dense) 
        self._init_weights(self.classifier.out_proj) 

    def _init_weights(self, module, initialize_range=0.02):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initialize_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output = outputs[0]
        sequence_output = outputs.last_hidden_state
        sentemb = sequence_output[:,0,:] # take <s> token (equiv. to [CLS])
        logits = self.classifier(sentemb)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                # loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class FrameBinaryClassification(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.config = config

        # self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.roberta = AutoModel(config, add_pooling_layer=False)
        self.classifier = ClassificationHead(config.hidden_size, config.hidden_dropout_prob, 1)
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sentemb = sequence_output[:,0,:] # take <s> token (equiv. to [CLS])
        logits = self.classifier(sentemb)

        loss = None
        if labels is not None:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
