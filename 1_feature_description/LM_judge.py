from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="YOUR_KEY",
)

LM_JUDGE_PROMPT = """Does this feature description accurately describe when this feature activates?
Rate on a scale of:
- 1 = completely unrelated to expected
- 2 = mostly unrelated
- 3 = somewhat related
- 4 = related and fairly similar
- 5 = same as expected, or highly similar (treat this as a correct match)

If unsure between 4 and 5, choose 5.

Examples:

Predicted: mentions of cooking recipes
Expected: references to financial transactions
Correct rating: 1

Predicted: mentions of dogs and cats
Expected: references to farm animals
Correct rating: 2

Predicted: mentions of sunny weather and rain
Expected: references to climate conditions
Correct rating: 3

Predicted: mentions of jazz musicians and concerts
Expected: references to music
Correct rating: 4

Predicted: mentions of Shakespeare's plays
Expected: references to works by Shakespeare
Correct rating: 5

Now rate the following pair:

Predicted: {predicted_label}
Expected: {expected_label}

Return a number from 1 to 5 and nothing else.
"""

def lm_judge_score(predicted: str, expected: str) -> int:
    prompt = LM_JUDGE_PROMPT.format(
        predicted_label=predicted.strip(),
        expected_label=expected.strip()
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,     # <-- use `input`, not `messages`
        temperature=0
    )
    # print(resp)
    text = resp.output_text.strip()
    try:
        return int(text)
    except ValueError:
        raise RuntimeError(f"Judge returned invalid output: {text}")


if __name__ == "__main__":
    pred = "phrases about similarity and comparison"
    gold = "words indicating equivalence or similarity"

    score = lm_judge_score(pred, gold)
    print(score)
