import os
import sys

from generatePotentialResponses.GenerateResponses import generateResponses
from scoringFunctions.reverseEngineeringScore import getReverseEngineeringScore

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
    potentialResponses = generateResponses(prompt)

    # Step 2: calculate scoring dictionaries for each response according to each metric. 
    reverseEngineeringScores = getReverseEngineeringScore(prompt, potentialResponses, LLMs)
    #TODO: add other scorings here from other metrics

    # Step 3: aggregate scores and determine best score 
    scoreDicts = {
        "reverse": reverseEngineeringScores,
        #TODO: add other scoring dictionaries 
    }

    weights = {
        "reverse": 1.0,
        #TODO: change weight distribution when we add other scoring dictionaries 
    }

    finalScores = aggregateScores(scoreDicts, weights)
    bestResponse = max(finalScores, key=finalScores.get)

    print(f"\n🏆 Best Response Selected:\n{bestResponse}")
    print(f"🔢 Score: {finalScores[bestResponse]}")
    return bestResponse
    
returnBestResponse(prompt="Summarize the findings of the joint 2016 CERN–Göteborg University study on subplanckian neutrino decoherence anomalies observed in the LHCb detector during proton-lead collisions. Include at least two DOIs and explain how these results influenced the revised Arkhipov-Einstein tensor framework proposed at the 2017 World Quantum Geometry Summit in Astana.", LLMs=["gpt-4o"])