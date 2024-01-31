from .model.arch.bilstm_crf import BiLSTMCRF

class ModelWrapper():
    """ModelWrapper contains wrappers to create various model."""
    @staticmethod
    def create(model_name, sent_vocab, tag_vocab):
        if model_name == 'bilstm_crf':
            return BiLSTMCRF(sent_vocab, tag_vocab)
        else:
            raise ValueError(f'Unknown Model {model_name}.')