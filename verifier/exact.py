from .checks import decimal_problem, qualifying_problem, result, scored
from .rule import parse_decimal


def verify_input(value):
    problem = decimal_problem(value.threshold, "threshold") or decimal_problem(value.min_recall, "min_recall")
    if problem:
        return result(False, problem)
    if value.budget < 1:
        return result(False, "budget must be at least 1")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if (output_value.dataset, output_value.threshold) != (input_value.dataset, input_value.threshold):
        return result(False, "output is for another dataset or threshold than the input")
    keywords = set(output_value.keywords)
    if len(keywords) != len(output_value.keywords) or len(keywords) < 2:
        return result(False, "the keywords must be at least 2 and unique")
    pairs = [(pair.a, pair.b) for pair in output_value.pairs]
    if pairs != sorted(set(pairs)):
        return result(False, "the pairs must be unique and sorted by a then b")
    threshold = parse_decimal(input_value.threshold)
    for pair in output_value.pairs:
        if pair.a not in keywords or pair.b not in keywords:
            return result(False, f"pair ({pair.a!r}, {pair.b!r}) names a keyword the dataset does not hold")
        problem = qualifying_problem(scored(pair), threshold)
        if problem:
            return result(False, problem)
    return result(True, "every qualifying pair is listed", pairs=len(pairs))
