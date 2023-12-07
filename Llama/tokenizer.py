import os
from sentencepiece import SentencePieceProcessor
from typing import List


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
         编码器
        :param s: 需要编码的字符串
        :param bos: 是否识别字符前的 回车，缩进，空格 等空白字符
        :param eos: 是否识别字符后的 回车，缩进，空格 等空白字符
        :return: 返回一个list，编码后的结果
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
         解码器
        :param t:
        :return: 返回解码后的字符串
        """
        return self.sp_model.decode(t)

    def getVocabSize(self):
        return self.n_words