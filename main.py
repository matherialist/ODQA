from DocQA import DocQA


if __name__ == '__main__':
    bot = DocQA('configs/ru_ranker_tfidf_wiki.json')
    bot.run('tg_token')
