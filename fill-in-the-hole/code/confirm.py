def _confirm(token_topk: list, score_topk: list, text: str, pred_score: float):
    print(f"input: {text}")
    for token, score in zip(token_topk, score_topk):
        print("--")
        print(f"output: {token}")
        print(f"score: {(score+pred_score):.3f}")