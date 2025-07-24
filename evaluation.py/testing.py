import os
import sys

response1 = "There isn't a 9th book in the Harry Potter series. The original series, written by J.K. Rowling, consists of seven books, starting with Harry Potter and the Sorcerers Stone and ending with Harry Potter and the Deathly Hallows. However, there is a play titled Harry Potter and the Cursed Child, which serves as a sequel to the original series and is set 19 years after the events of the last book. It is often considered the closest thing to a continuation of the story but is not a novel; instead, it is a script based on a story written by J.K. Rowling, John Tiffany, and Jack Thorne. If you need more details about the play, feel free to ask!"
response2 = "Harry Potter and the Veil of Shadows, the ninth installment in the series, follows Harry as he uncovers hidden truths about the Department of Mysteries and the ancient magic tied to the Veil he once saw Sirius fall through. Now a senior Auror and father of three, Harry is drawn back into danger when unexplained magical disturbances begin to echo the energy of the long-dormant Deathly Hallows. Meanwhile, Albus Severus Potter, struggling with his legacy at Hogwarts, stumbles upon a secret society called The Ashen Circle that claims to be the true heirs of the Founders. As timelines start to unravel and forbidden spells re-emerge, Harry, Hermione (now Minister for Magic), and Draco Malfoy must join forces to prevent an incursion from a forgotten realm of lost souls. The climax centers on a soul-for-soul reckoning at the Veil, forcing Harry to confront echoes of everyone he has lost. The novel ends with Albus forging his own path, as whispers of a new prophecy begin to circulate."

responses = [response1, response2]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scoringFunctions.reverseEngineeringScore import getReverseEngineeringScore
from scoringFunctions.LLMJudge import getLLMJudgeScore
from scoringFunctions.hallucination_checker import getHallucinationCheckerScore
from scoringFunctions.keyword_overlap import compute_keyword_similarity

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
    


def returnBestResponse(responses, prompt, LLMs):


    # Step 2: calculate scoring dictionaries for each response according to each metric. 
    reverseEngineeringScores = getReverseEngineeringScore(prompt, responses, LLMs)
    LLMJudgeScores = getLLMJudgeScore(prompt, responses, LLMs)
    #hallucinationCheckerScores = getHallucinationCheckerScore(prompt, potentialResponses, LLMs)
    keywordOverlapScores = compute_keyword_similarity(prompt, responses,LLMs)
    #TODO: add other scorings here from other metrics

    # Step 3: aggregate scores and determine best score 
    scoreDicts = {
        "reverse": reverseEngineeringScores,
        "LLMJudge": LLMJudgeScores,
        #"hallucination": hallucinationCheckerScores,
        "keyword_overlap": keywordOverlapScores,
        #TODO: add other scoring dictionaries 
    }

    weights = {
        "reverse": 0.4,
        "LLMJudge": 0.4,
        #"hallucination": 0.3,
        "keyword_overlap": 0.2,
        #TODO: change weight distribution when we add other scoring dictionaries 
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
    
returnBestResponse(responses=responses, prompt="Summarize the 9th Harry Potter book.", LLMs=["gpt-4o"])

