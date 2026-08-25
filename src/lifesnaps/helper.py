"""Helper functions."""


def try_convert_to_numeric(value):
    """Tries to convert a string to numeric if possible."""
    try:
        return float(value) if "." in value else int(value)
    except (ValueError, TypeError):
        return value


def inverse_scores(score, min, max):
    """
    Inverts questionnaire likert scores.
    """
    return max - score + min


def define_stage_of_change(response):
    """Defines the stage of change according to Stages and Processes of Change Questionnaire."""
    if (
        response
        == "No, and I do not intend to do regular physical activity in the next 6 months."
    ):
        return "Precontemplation"
    if (
        response
        == "No, but I intend to do regular physical activity in the next 6 months."
    ):
        return "Contemplation"
    if (
        response
        == "No, but I intend to do regular physical activity in the next 30 days."
    ):
        return "Preparation"
    if (
        response
        == "Yes, I have been doing physical activity regularly, but for less than 6 months."
    ):
        return "Action"
    return "Maintenance"


def convert_5_to_4_likert(x):
    """Converts 5-likert to 4-likert scale."""
    x = (4 - 1) * (x - 1) / (5 - 1) + 1
    dec = 0
    num = str(x)[: str(x).index(".") + dec + 2]
    if num[-1] >= "5":
        a = num[: -2 - (not dec)]  # integer part
        b = int(num[-2 - (not dec)]) + 1  # decimal part
        return float(a) + b ** (-dec + 1) if a and b == 10 else float(a + str(b))
    return float(num[:-1])
