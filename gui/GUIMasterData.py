class GUIMasterData:
    data_changed: bool = False
    data_type: str = ""
    do_not_update: list[str] = ["do_not_update", "data_type"]

    def __init__(self) -> None:
        self.data_type = str(type(self)).split(".")[-1][:-2]

    def update(self, input) -> None:
        to_update: list[str] = []

        something_todo = getattr(input, "data_changed", None)
        if (something_todo is None) or (something_todo is False):
            return

        for vars in dir(self):
            if vars.startswith("__") is False:
                if not callable(getattr(self, vars)):
                    if (vars in self.do_not_update) is False:
                        to_update.append(vars)

        input_name = getattr(input, "data_type", None)
        if (input_name is not None) and (input_name == self.data_type):
            for vars in to_update:
                data_in = getattr(input, vars, None)
                setattr(self, vars, data_in)
