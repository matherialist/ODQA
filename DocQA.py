import webbrowser
from deeppavlov.core.common.file import read_json
from deeppavlov import configs, train_model
from deeppavlov.core.commands.infer import build_model
from deeppavlov.utils.telegram import interact_model_by_telegram


class DocQA:

    def __init__(self, model_config):
        model_config = read_json(model_config)
        train_model(model_config)
        build_model(configs.squad.multi_squad_noans_infer, download=True)

    def run(self, token):
        interact_model_by_telegram(configs.odqa.en_odqa_infer_wiki, token)
        webbrowser.open_new_tab('https://web.telegram.org/#/im?p=@TPresentationBot')
