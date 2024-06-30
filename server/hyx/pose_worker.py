from multiprocessing import Process

class PoseWorker(Process):
    def __init__(self, model_name, in_queue, out_queue, model):
        super().__init__(name=f'{model_name}Processor')
        self.model_name = model_name
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model = model

    def run(self):
        while True:
            frame = self.in_queue.get()
            result = self.model.inference(frame)
            self.out_queue.put(result)
