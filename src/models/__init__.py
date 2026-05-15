from .transformer_model import Seq2Seq as TransformerModel


# Import user defined models
from .user_defined_transformer import TransformerModel as TransformerLayerwise
from .user_defined_transformer import TransformerThreeLayersMemoryConnect
from .user_defined_transformer import TransformerThreeLayersMemory

def get_model(config):
    """
    模型工廠函數：根據設定檔返回對應的模型實例。
    """
    model_name = config['model']['name']
    model_params = config['model']
    
    # 獲取輸入輸出的維度 (這些參數是通用的)
    # 注意：GRUExact 的維度可能在 config['variables_num'] 裡，需要在 GRUExact 內部處理
    # 這裡只提取通用參數傳給舊模型接口
    num_en_input = config['data'].get('num_en_input', 10)
    num_de_input = config['data'].get('num_de_input', 2)
    num_output = config['data'].get('num_output', 2)
    
    if model_name == 'gru_exact':
        return GRUExact(config)
    elif model_name == 'gru':
        return GRUModel(
            num_en_input=num_en_input,
            num_de_input=num_de_input,
            num_output=num_output,
            embedding_dim=model_params['embedding_dim'],
            hidden_dim=model_params['hidden_dim'],
            n_layers=model_params['n_layers']
        )
    elif model_name == 'transformer':
        return TransformerModel(
            num_en_input=num_en_input,
            num_de_input=num_de_input,
            num_output=num_output,
            embedding_dim=model_params['embedding_dim'],
            hidden_dim=model_params['hidden_dim'],
            n_layers=model_params['n_layers'],
            n_heads=model_params.get('num_heads', 1),
            dropout=model_params.get('dropout', 0.1)
        )
    
    # User Defined Models
    elif model_name == 'transformer_layerwise':
        return TransformerLayerwise(
            num_input=num_en_input,
            num_output=num_output,
            num_embs=model_params['embedding_dim'],
            intermediate_dim=model_params['hidden_dim'] * model_params.get('ffn_expansion', 2),
            num_heads=model_params.get('num_heads', 8), # Default to 8 if not in config
            num_layers=model_params['n_layers'],
            activation_func=model_params.get('activation', 'tanh'),
            dropout=model_params.get('dropout', 0.1)
        )
    elif model_name == 'transformer_memory_connect':
        return TransformerThreeLayersMemoryConnect(
            num_input=num_en_input,
            num_output=num_output,
            num_embs=model_params['embedding_dim'],
            intermediate_dim=model_params['hidden_dim'] * 4,
            num_heads=model_params.get('num_heads', 8),
            w=model_params.get('w', 18),
            activation_func=model_params.get('activation', 'tanh')
        )
    elif model_name == 'transformer_memory':
        return TransformerThreeLayersMemory(
            num_input=num_en_input,
            num_output=num_output,
            num_embs=model_params['embedding_dim'],
            intermediate_dim=model_params['hidden_dim'] * 4,
            num_heads=model_params.get('num_heads', 8),
            w=model_params.get('w', 18),
            activation_func=model_params.get('activation', 'tanh')
        )
    else:
        raise ValueError(f"未知的模型名稱: {model_name}")