import mesa


class Fish(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, catch_rate: float = 0.5):
        super().__init__(unique_id, model)
        self.catch_rate = catch_rate

    def step(self):
        pass
