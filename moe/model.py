import torch
import torch.nn as nn

from cgcnn.model import CrystalGraphConvNet
from cgcnn.utils import initialize_kwargs


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


def get_extractor(checkpoint_path=None, layer_to_extract_from='conv',
                  model_kwargs=None, device='cpu'):
    """
    Args:
        checkpoint_path (str): Path to checkpoint containing the model state
            dict.
        layer_to_extract_from (str): If 'conv', extract the features from the
            last convolutional layer. If 'fc', extract the features from the
            penultimate fully connected layer. If 'conv-2', extract features
            from the second to last convolutional layer.
        task_name (str): Name of the task. Required if the model is multi-headed
            so we know which head to use as the feature extractor.
        model_kwargs (dict): CrystalGraphConvNet kwargs.

    Returns:
        Model with all layers after the feature extraction layer set to the
        identity. (torch.nn.module)
    """
    if model_kwargs is None:
        model_kwargs, _ = initialize_kwargs()
    model = CrystalGraphConvNet(**model_kwargs)

    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device(device))
        except RuntimeError:
            raise RuntimeError(
                'Either your PyTorch version is too old, or the'
                'following path does not exist: ' + checkpoint_path)

        best_model_state_dict = checkpoint['state_dict']
        model.load_state_dict(best_model_state_dict)
    else:
        import warnings
        warnings.warn('checkpoint_path is None. Learning a backbone from '
                      'scratch.')

    # set layers after feature extractor layer to identity function
    if layer_to_extract_from == 'conv':
        model.conv_to_fc_softplus = Identity()
        model.conv_to_fc = Identity()
        model.dropout = Identity()
        model.fcs = nn.ModuleList([Identity()])
        model.softpluses = nn.ModuleList([Identity()])
        model.logsoftmax = Identity()
        model.fc_out = nn.Softplus()    # Softplus

    elif layer_to_extract_from == 'conv-2':
        # remove the last conv but still apply pooling
        model.convs[-1] = Identity()

        model.conv_to_fc_softplus = Identity()
        model.conv_to_fc = Identity()
        model.dropout = Identity()
        model.fcs = nn.ModuleList([Identity()])
        model.softpluses = nn.ModuleList([Identity()])
        model.logsoftmax = Identity()
        model.fc_out = nn.Softplus()    # Softplus

    elif layer_to_extract_from == 'first_fc':
        model.dropout = Identity()
        model.fcs = nn.ModuleList([Identity()])
        model.softpluses = nn.ModuleList([Identity()])
        model.logsoftmax = Identity()
        model.fc_out = Identity()

    elif layer_to_extract_from == 'penultimate_fc':
        model.fc_out = Identity()
    else:
        raise AttributeError(
            'layer_to_extract_from must be \'conv\', \'first_fc\', or '
            '\'penultimate_fc\'.')
    return model


class EnsemblePredictor(nn.Module):
    def __init__(self, num_predictions_to_ensemble):
        super(EnsemblePredictor, self).__init__()
        self.num_predictions_to_ensemble = num_predictions_to_ensemble
        self.weights = nn.Parameter(torch.ones(num_predictions_to_ensemble),
                                    requires_grad=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, predictions):
        normalized_weights = self.softmax(self.weights)

        out = torch.sum(normalized_weights * predictions, dim=1)
        return out


class MixtureOfExtractors(nn.Module):
    def __init__(self, extractors, option='pairwise_TL', k_experts=0):
        """
        If 'option' is 'pairwise_TL', let 'extractors' only contain a single
        extractor.

        Args:
            extractors (list of nn.Module): Extractors which all have output
                of the same dimensionality.
            option (str):
        """
        super(MixtureOfExtractors, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.option = option
        self.k_experts = k_experts

        if self.option == 'add_k':
            assert 0 < self.k_experts <= len(extractors)

        self.scaling_params = nn.Parameter(
            torch.ones(len(self.extractors)),
            requires_grad=option != 'pairwise_TL' and option != 'ensemble')
        self.softmax = nn.Softmax(dim=0)

    def forward(self, structure):
        out = self.scaling_params[0] * self.extractors[0](*structure)

        if self.option == 'pairwise_TL':
            return out
        elif self.option == 'add_k':
            top_k_values, top_k_indices = torch.topk(
                self.scaling_params, self.k_experts)
            top_k_probabilities = self.softmax(top_k_values)

            top_prob, top_idx = top_k_probabilities[0], top_k_indices[0]
            out = top_prob * self.extractors[top_idx](*structure)

            for i in range(self.k_experts-1):

                prob, idx = top_k_probabilities[i+1], top_k_indices[i+1]
                out = out + prob * self.extractors[idx](*structure)

            # sparsified backbone weights of shape: (len(self.extractors),)
            backbone_scores = torch.zeros(
                len(self.extractors), device=self.scaling_params.device).scatter(
                0, top_k_indices, top_k_probabilities)

            return out, backbone_scores

        elif self.option == 'concat':
            for i in range(len(self.extractors) - 1):
                scaled_backbone_to_concat = \
                    self.scaling_params[i+1] * self.extractors[i+1](*structure)
                out = torch.cat((out, scaled_backbone_to_concat), dim=1)
            return out
        else:
            raise NotImplementedError

    def non_extractor_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad and 'extractors' not in n:
                yield p


class MultilayerPerceptronHead(nn.Module):
    def __init__(
            self, num_layers, input_dim, option='pairwise_TL', hidden_dim=None):
        super(MultilayerPerceptronHead, self).__init__()
        self.softplus = nn.Softplus()
        self.num_layers = num_layers

        if hidden_dim is None:
            hidden_dim = 32 if not option == 'concat' else 32 * 3

        layers = []
        if self.num_layers > 1:
            layers.append(nn.Linear(input_dim, hidden_dim))
        if self.num_layers > 2:
            for _ in range(self.num_layers-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        if num_layers == 1:
            self.out_layer = nn.Linear(input_dim, 1)
        else:
            self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, feature):
        out = feature
        for fc in self.layers:
            out = self.softplus(fc(out))
        out = self.out_layer(out)
        return out


class MultiheadedMixtureOfExpertsModel(nn.Module):
    def __init__(self, num_pseudo_attention_heads, backbones,
                 num_out_layers, backbone_feature_dim, k_experts=1):
        super(MultiheadedMixtureOfExpertsModel, self).__init__()
        self.num_pseudo_attention_heads = num_pseudo_attention_heads
        self.k_experts_per_head = k_experts
        self.backbone_feature_dim = backbone_feature_dim  # dimensionality of a single backbone's output

        self.pseudo_attention_heads = nn.ModuleList()
        for _ in range(num_pseudo_attention_heads):
            pseudo_attention_head = MixtureOfExtractors(
                backbones, option='add_k', k_experts=self.k_experts_per_head)
            self.pseudo_attention_heads.append(pseudo_attention_head)

        self.out_layer = MultilayerPerceptronHead(
            num_out_layers,
            input_dim=backbone_feature_dim * num_pseudo_attention_heads,
            option='pairwise_TL', hidden_dim=32*num_pseudo_attention_heads)

    def forward(self, structure):
        n_structures = len(structure[-1])
        head_outputs = []
        score_lst = []

        out, extractor_scores = self.pseudo_attention_heads[0](structure)
        head_outputs.append(out)
        score_lst.append(extractor_scores)

        for i in range(self.num_pseudo_attention_heads-1):
            out, extractor_scores = self.pseudo_attention_heads[i+1](structure)

            head_outputs.append(out)
            score_lst.append(extractor_scores)

        # batch_size, backbone_feature_size * num_pseudo_attention_heads
        multihead_feature = torch.stack(
            head_outputs, dim=-1).view(n_structures, -1)

        # shape: (n_extractors, num_pseudo_attention_heads)
        score_lst = torch.stack(score_lst, dim=-1)

        # Use loss regularizer to avoid collapse to a single set of extractors
        # when self.num_pseudo_attention_heads > 1, and to a single extractor
        # when self.num_pseudo_attentio_heads == 1
        loss_regularizer = torch.pow(
            torch.norm(
                torch.transpose(score_lst, 0, 1) @ score_lst -
                torch.eye(self.num_pseudo_attention_heads,
                          device=score_lst.device)), 2)

        out = self.out_layer(multihead_feature)
        return out, loss_regularizer

    def non_extractor_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad and 'extractors' not in n:
                yield p
