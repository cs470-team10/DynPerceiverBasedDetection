import re

def file_name(config_entry, name = "coco_analysis_accuarcy", posfix=".csv"):
    accuarcy = re.sub("[.]", "_", str(config_entry["accuarcy"]))
    flops = re.sub("[.]", "_", str(config_entry["flops"]))
    return f"{config_entry['index']}_{name}__accuarcy_{accuarcy}__flops_{flops}{posfix}"

def formatting_config_entry(config_entry, path = ''):
    path = f", be saving in {path}" if path != '' else ""
    return f"[{config_entry['index']}] Accuarcy: {config_entry['accuarcy']}%, Flops: {config_entry['flops']}{path}"