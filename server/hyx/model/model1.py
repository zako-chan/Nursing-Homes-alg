def load_model():
    print("Loading model 1")
    class Model:
        def inference(self, frame):
            return "Processed by model 1"
    return Model()
