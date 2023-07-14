import mesa


class Fish(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, p_catch: float = 0):
        """
        Fish agent
        :param unique_id:
        :param model:
        :param p_catch: Probability of being caught
        """
        super().__init__(unique_id, model)
        self.catch_rate = p_catch

    def step(self):
        pass
