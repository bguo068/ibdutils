class CheckableParams:
    """A class that checks attributes values and allows autocompletion"""

    def __init__(self) -> None:
        """
        Must be called at the end of child class __init__ method
        - save attributes names and default values from the child class
        - after save defaults from child class init function, clear all attributes.
        - create a dict to indicate whehter an attribute has been set after clear out
        """

        # after setting attributes
        self.__defaults__ = self.as_dict()
        self.__unset_attrib__ = {k: True for k in self.__defaults__.keys()}
        for k in self.__defaults__.keys():
            delattr(self, k)

    def __setattr__(self, name, value):
        """
        - when an attribute was set, check if the attribute are allowed;
        - update indicator dict to mark this attribute has been set

        """
        if (
            not name.startswith("__")
            and not callable(name)
            and hasattr(self, "__defaults__")
            and hasattr(self, "__unset_attrib__")
        ):
            self.__unset_attrib__[name] = False
            if name not in self.__defaults__.keys():
                raise Exception(f"{name} attribute is not allowed")
        super().__setattr__(name, value)

    def as_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(k)
        }

    def fill_defaults(self):
        for k, v in self.__defaults__.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)

    def get_unset_params(self):
        return [k for k, v in self.__unset_attrib__.items() if v]


if __name__ == "__main__":

    class ExampleParams(CheckableParams):
        def __init__(self) -> None:
            self.a = 100
            self.b = 1
            super().__init__()

    p = ExampleParams()
    print(p.as_dict())
    p.a = 100
    try:
        print(p.b)
    except:
        print("cannot access 'b' attribute")

    print(p.as_dict())
    print(p.get_unset_params())
    p.fill_defaults()
    print(p.as_dict())
    try:
        p.c = 1000
    except:
        print("cannot assign 'c' attribute")

    p.d = 100
