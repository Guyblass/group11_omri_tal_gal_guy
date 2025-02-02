import logging
import json

import fire


from datasets import Dataset

from tools.bot import load_bot

logger = logging.getLogger(__name__)


def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str, metrics: list) -> dict:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    from ragas import evaluate
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truths": [[ground_truth]],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)
    # Import ragas only after loading the environment variables inside load_bot()
    from ragas.metrics import (
        answer_correctness,
        answer_similarity,
        answer_relevancy,
    )
    from ragas.metrics.context_precision import context_relevancy
    metrics = [
        answer_correctness,
        answer_similarity,
        answer_relevancy,
    ]

    num_of_tests = 0
    answer_correctness = 0
    answer_similarity = 0
    answer_relevancy = 0

    with open(testset_path, "r") as f:
        data = json.load(f)
        for elem in data:
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
            }
            output_context = bot.finbot_chain.chains[0].run(input_payload)
            response = bot.answer(**input_payload)
            logger.info("Question: %s", elem["question"])
            logger.info("Context: %s", [output_context])
            logger.info("Response: %s", response)
            score = evaluate_w_ragas(query=elem["question"], context=[output_context], output=response, ground_truth=elem["response"], metrics=metrics)
            logger.info("Score=%s", score)
            num_of_tests += 1
            answer_correctness += score["answer_correctness"]
            answer_similarity += score["answer_similarity"]
            answer_relevancy += score["answer_relevancy"]


    logger.info("answer_correctness:%s", str(answer_correctness/num_of_tests))
    logger.info("answer_similarity:%s", str(answer_similarity/num_of_tests))
    logger.info("answer_relevancy:%s", str(answer_relevancy/num_of_tests))

    return response


if __name__ == "__main__":
    fire.Fire(run_local)
