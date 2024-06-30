def load_model():
    print("Loading model 3")
    class Model:
        def inference(self, frame):
            return "Processed by model 3"
    return Model()
