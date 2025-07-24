import os
import sys

from generatePotentialResponses.GenerateResponses import generateResponses
from scoringFunctions.reverseEngineeringScore import getReverseEngineeringScore
from scoringFunctions.LLMJudge import getLLMJudgeScore
from scoringFunctions.hallucination_checker import getHallucinationCheckerScore
from scoringFunctions.keyword_overlap import compute_keyword_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def aggregateScores(scoreDicts, weights):
    """
    score_dicts: dict[str, dict[str, float]] — {metric_name: {response: score}}
    weights: dict[str, float] — {metric_name: weight}

    Returns:
        dict[str, float] — {response: aggregated_score}
    """
    aggregated = {}

    # Assumes all score dicts have the same keys (responses)
    for response in next(iter(scoreDicts.values())).keys():
        total = 0.0
        for metric, scores in scoreDicts.items():
            weight = weights.get(metric, 0)
            total += weight * scores[response]
        aggregated[response] = total

    return aggregated
    


def returnBestResponse(prompt, LLMs):

    # Step 1: generate potential responses. 
    potentialResponses = generateResponses(prompt, LLMs)

    # Step 2: calculate scoring dictionaries for each response according to each metric. 
    reverseEngineeringScores = getReverseEngineeringScore(prompt, potentialResponses, LLMs)
    LLMJudgeScores = getLLMJudgeScore(prompt, potentialResponses, LLMs)
    hallucinationCheckerScores = getHallucinationCheckerScore(prompt, potentialResponses, LLMs)
    keywordOverlapScores = compute_keyword_similarity(prompt, potentialResponses,LLMs)
    #TODO: add other scorings here from other metrics

    # Step 3: aggregate scores and determine best score 
    scoreDicts = {
        "reverse": reverseEngineeringScores,
        "LLMJudge": LLMJudgeScores,
        "hallucination": hallucinationCheckerScores,
        "keyword_overlap": keywordOverlapScores,
        #TODO: add other scoring dictionaries 
    }

    weights = {
        "reverse": 0.35,
        "LLMJudge": 0.25,
        "hallucination": 0.25,
        "keyword_overlap": 0.15,
    }

    finalScores = aggregateScores(scoreDicts, weights)
    
    # Step 4: Identify best and worst responses
    bestResponse = max(finalScores, key=finalScores.get)
    worstResponse = min(finalScores, key=finalScores.get)

    print(f"\n🏆 Best Response Selected:\n{bestResponse}")
    print(f"🔢 Best Score: {finalScores[bestResponse]}")

    print(f"\n🚫 Worst Response Identified:\n{worstResponse}")
    print(f"🔻 Worst Score: {finalScores[worstResponse]}")

    return {
        "best": {"response": bestResponse, "score": finalScores[bestResponse]},
        "worst": {"response": worstResponse, "score": finalScores[worstResponse]},
    }
    
returnBestResponse(prompt="HOw did elon musk develop chat gpt?", LLMs=["gpt-4o"])

