import yaml


def open_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def update_config_from_paths(args_dict,):
    if args_dict["yamldataset"] is not None :
        args_dict.update(open_yaml(args_dict["yamldataset"]))
    return args_dict
