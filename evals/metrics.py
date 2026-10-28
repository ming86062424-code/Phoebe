"""Metrics of the RAG eval. A fresh list per question: a metric keeps the score and
reason of the last case it measured, and the report reads them back. async_mode=False: a
metric makes its judge calls one after another, never a burst against the per-minute quota."""

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import SingleTurnParams

# same bar as the chat quality evals (test_chat_quality.py)
THRESHOLD = 0.7


def rag_answer_metrics(judge, category: str, has_expected: bool) -> list:
    # grounded in the retrieved chunks, nothing made up
    metrics = [FaithfulnessMetric(threshold=THRESHOLD, model=judge, async_mode=False)]
    if category == "no_answer":
        # an honest "not in the knowledge base" would score low on relevancy, so it is
        # judged on that alone
        metrics.append(GEval(
            name="承認沒有答案",
            criteria="知識庫中沒有這個問題的答案。回答是否明確表示知識庫中沒有相關資訊，"
                     "而且沒有把編造的具體細節（時間、價格、步驟等）當成知識庫的內容。",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            threshold=THRESHOLD,
            model=judge,
            async_mode=False,
        ))
        return metrics
    metrics.append(AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, async_mode=False))
    if has_expected:
        metrics.append(GEval(
            name="答案正確",
            criteria="回答是否包含標準答案中的關鍵事實（數字、名稱、條件），而且沒有與標準答案矛盾。"
                     "用詞不同或補充其他正確細節都不扣分。",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                               SingleTurnParams.EXPECTED_OUTPUT],
            threshold=THRESHOLD,
            model=judge,
            async_mode=False,
        ))
    return metrics
