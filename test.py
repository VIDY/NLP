import models.category.category_model

if __name__ == '__main__':

    # Load graph
    g = models.category.category_model.CategoryModel()
    print("Graph loaded")
    print(str(g.eval([{"text": "Actual weather in Farafra: 19°C, Clear - http://www.weatherio.com/egypt/farafra&nbsp; Farafra Egypt"},{"text": "Actual weather in Farafra: 19°C, Clear - http://www.weatherio.com/egypt/farafra&nbsp; Farafra Egypt"}])))

