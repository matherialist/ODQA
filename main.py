from deeppavlov.core.common.file import read_json
from deeppavlov import configs, train_model
from deeppavlov.core.commands.infer import build_model
from deeppavlov.utils.telegram import interact_model_by_telegram


model_config = read_json("configs/ru_ranker_tfidf_wiki.json")
#model_config = read_json(configs.doc_retrieval.en_ranker_tfidf_wiki)
#model_config["dataset_reader"]["data_path"] = "~/Desktop/ODQA/train_data"
#model_config["dataset_reader"]["dataset_format"] = "txt"

ranker = train_model(model_config)
ranker(['cerebellum'])

squad = build_model(configs.squad.multi_squad_ru_retr_noans_rubert_infer, download=True)

odqa = build_model("/configs/ru_odqa_infer_wiki_rubert_noans.json", download=False)
answers = odqa(["what is tuberculosis?", "how should I take antibiotics?"])

print(answers)

#interact_model_by_telegram(model_config=configs.odqa.en_odqa_infer_wiki, token='')
