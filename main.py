from deeppavlov.core.common.file import read_json
from deeppavlov import configs, train_model
from deeppavlov.core.commands.infer import build_model


if __name__ == '__main__':
    model_config = read_json("configs/en_ranker_tfidf_wiki.json")

    ranker = train_model(model_config)

    squad = build_model(configs.squad.multi_squad_noans_infer, download=True)

    odqa = build_model("configs/en_odqa_infer_wiki.json", download=False)
    answers = odqa(["what is tuberculosis?"])

    print(answers)
