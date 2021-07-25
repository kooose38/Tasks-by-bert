from model import BertForMaskedLM_ 
from tokenizer import tokenizer_
from data import random_input
import torch 
from confirm import _confirm
import numpy as np 
import warnings 
warnings.filterwarnings("ignore")

class PredictFromMaskedSentence:
    def __init__(self):
#         self.text =  random_input(text)
        self.tokenizer = tokenizer_()
        self.i = 0
    def _predict_mask_topk(self, text: str, num_topk: int, pred_score: float):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        scores = BertForMaskedLM_(input_ids)
        # [MASK]の最初のindexを求める
        mask_position = input_ids[0].tolist().index(4)
        # 32000次元からnum_topk数だけindexの取得
        topk = scores[0, mask_position].topk(num_topk)
        ids_topk = topk.indices 
        # indexからbert辞書が位置する単語を取得
        # ["予測単語１", "予測単語２", "予測単語３"]
        tokens_topk = self.tokenizer.convert_ids_to_tokens(ids_topk)
        scores_topk = topk.values.cpu().numpy()
        
        text_topk = []
        # 文章中の[MASK]を予測単語に置き換える
        for token in tokens_topk:
            token = token.replace("##", "")
            text_topk.append(text.replace("[MASK]", token, 1))
        x = ""
        if self.i == 0:
            x = str(input(f"Do you want to output the predicted top {num_topk} patterns? (yes/no)"))
        if (x in ["yes", "y", "Y"]) | (self.i == 1):
            self.i = 1
            _confirm(text_topk, list(scores_topk), text, pred_score)
        return text_topk, scores_topk
    def greedy_prediction(self, text:str, num_mask: int, num_topk: int):
        """
        貪欲法による穴埋め解決
        [MASK]が複数ある場合に、最もスコアの高かった最初の穴埋めを行い次の入力として予測を繰り返し行います。
        予測する毎のスコアを基準とするため多くの[MASK]には対応が困難になります。
        """
        text = random_input(text, num_mask)
        for _ in range(text.count("[MASK]")):
            print(f"<<{_+1} layers>>")
            text = self._predict_mask_topk(text, num_topk, 0.0)[0][0]
            
        print(f"final predicted values: {text}")
    def beam_search(self, text: str, num_mask: int, num_topk: int):
        text = random_input(text, num_mask)
        scores_topk = np.array([0])
        text_topk = [text]
        for _ in range(text.count("[MASK]")):
            print(f"<<{_+1} layers>>")
            text_candidate, score_candidate = [], []
            
            for text_mask, score in zip(text_topk, scores_topk):
                text_topk_inner, scores_topk_inner = self._predict_mask_topk(
                    text_mask, num_topk, score
                )
                text_candidate.extend(text_topk_inner) # 予測された文章
                score_candidate.append(score+scores_topk_inner) # 文章ごとの評価点
                
            score_candidate = np.hstack(score_candidate)
            # スコアから合計が高いindexの上位num_topk分だけ取得する
            idx_list = score_candidate.argsort()[::-1][:num_topk]
            # 次のループに回す
            text_topk = [text_candidate[idx] for idx in idx_list] # 評価が高い文章とスコア
            scores_topk = [score_candidate[idx] for idx in idx_list]
            
        print(f"final predicted values: {text_topk[0]}")