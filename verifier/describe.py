from .checks import HYPERPLANES, result


def verify_input(value):
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.dataset != input_value.dataset:
        return result(False, f"output describes {output_value.dataset.value}, not {input_value.dataset.value}")
    if output_value.keywords < 2 or output_value.dimension < 1:
        return result(False, "a dataset needs at least 2 keywords and vectors of at least 1 entry")
    if output_value.hyperplanes != HYPERPLANES:
        return result(False, f"the plugin bundles {HYPERPLANES} hyperplanes, not {output_value.hyperplanes}")
    if output_value.pairs != output_value.keywords * (output_value.keywords - 1) // 2:
        return result(False, f"{output_value.keywords} keywords make {output_value.keywords} choose 2 pairs")
    return result(True, "the dataset is described", keywords=output_value.keywords, pairs=output_value.pairs)
