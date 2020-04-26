import os
from configparser import ConfigParser

CONFIG_FILE = os.path.abspath(os.path.dirname(__file__)) + '/config.ini'
PARSER = ConfigParser()
PARSER.read(CONFIG_FILE)


def get_config(section, key):
	return PARSER.get(section, key)


if __name__ == '__main__':
	print(get_config("model", "c3d"))
