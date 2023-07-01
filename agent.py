import mesa


class InfotaxisAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model):
        super().__init__(unique_id, model)
