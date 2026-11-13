"""Prints the two judging prompts the recorded LLM was given, verbatim from the recording's manifest. The prose's point rests on the rules printed at the end of the guidelines prompt, after the word counts."""
from lib.recorded import manifest

recording = manifest("llm-judgments")             # the recording's model, settings and both prompts, verbatim
short, guidelines = recording["prompts"]["short"], recording["prompts"]["guidelines"]

print("model:", recording["model"])
print("what the model saw:", recording["sees"])
print("words in the short prompt:", len(short.split()))
print("words in the guidelines prompt:", len(guidelines.split()))
print()
print(guidelines)
