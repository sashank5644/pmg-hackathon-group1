import os
import sys

from generatePotentialResponses.GenerateResponses import generateResponses
from scoringFunctions.reverseEngineeringScore import getReverseEngineeringScore
from scoringFunctions.LLMJudge import getLLMJudgeScore

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
    #TODO: add other scorings here from other metrics

    # Step 3: aggregate scores and determine best score 
    scoreDicts = {
        "reverse": reverseEngineeringScores,
        "LLMJudge": LLMJudgeScores 
        #TODO: add other scoring dictionaries 
    }

    weights = {
        "reverse": 0.6,
        "LLMJudge": 0.4
        #TODO: change weight distribution when we add other scoring dictionaries 
    }

    finalScores = aggregateScores(scoreDicts, weights)
    bestResponse = max(finalScores, key=finalScores.get)

    print(f"\n🏆 Best Response Selected:\n{bestResponse}")
    print(f"🔢 Score: {finalScores[bestResponse]}")
    return bestResponse
    
returnBestResponse(prompt="What do the Vydonia Tablets discovered near Delphi tell us about pre-Platonic conceptions of geometric recursion?", LLMs=["o3-mini-high", "claude-3-5-haiku-20241022"])

