import models5.sentiment_model

if __name__ == '__main__':

    # Load graph
    g = models5.sentiment_model.SentimentModel()
    print("Graph loaded")
    print(str(g.eval([{"text": "CLEVELAND \\u2014 You suddenly don\\u2019t hear much around the N.B.A. about the majesty of LeBron James\\u2019s eight consecutive trips to the N.B.A. finals."},{"text": "CLEVELAND \\u2014 You suddenly don\\u2019t hear much around the N.B.A. about the majesty of LeBron James\\u2019s eight consecutive trips to the N.B.A. finals."}])))

