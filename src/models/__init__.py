from .gru_model import Seq2Seq as GRUModel
from .transformer_model import Seq2Seq as TransformerModel

def get_model(config):
    """
    模型工廠函數：根據設定檔返回對應的模型實例。
    """
    model_name = config['model']['name']
    model_params = config['model']
    
    # 獲取輸入輸出的維度 (這些參數是通用的)
    num_en_input = config['data']['num_en_input']
    num_de_input = config['data']['num_de_input']
    num_output = config['data']['num_output']
    
    if model_name == 'gru':
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
            n_layers=model_params['n_layers']
        )
    else:
        raise ValueError(f"未知的模型名稱: {model_name}")