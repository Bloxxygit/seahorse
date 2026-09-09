"""Run a no-dependency SeaGlass AI pipeline sanity check."""

from config import SETTINGS
from dataset import Example
from evaluation import Evaluator
from inference import InferenceService
from model import PlaceholderModel
from tokenizer import CharacterTokenizer
from training import TrainingLoop


def main() -> None:
    tokenizer = CharacterTokenizer()
    model = PlaceholderModel(SETTINGS["model"].vocabulary_size)
    examples = [Example("SeaGlass AI")]

    training_report = TrainingLoop().run(model, examples)
    generated = InferenceService(model, tokenizer).generate_one_token("SeaGlass")
    metrics = Evaluator().evaluate(model, examples)

    print("SeaGlass AI sanity check passed")
    print(f"training: {training_report}")
    print(f"generated token: {generated!r}")
    print(f"evaluation: {metrics}")


if __name__ == "__main__":
    main()
