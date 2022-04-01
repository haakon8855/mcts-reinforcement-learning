"""haakon8855"""

import configparser


class Config:
    """
    Fetches the config from a file and returns it in json/py-dict format.
    """
    @staticmethod
    def get_config(config_file: str):
        """
        Fetches the config from the given configuration file path and returns
        a json-object/python-dictionary containing the values.
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
