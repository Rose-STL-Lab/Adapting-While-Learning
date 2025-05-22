import numpy as np
import random
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def generate_number_choice(correct_answer):
    correct_position = random.randint(0, 3)
    correct_option = chr(65 + correct_position)

    options = [None] * 4
    options[correct_position] = correct_answer

    deviation_range = min(abs(correct_answer), 1)

    if isinstance(correct_answer, int):
        min_gap = max(1, int(0.1 * abs(correct_answer)))
    else:
        decimal_places = len(str(correct_answer).split(".")[-1])
        min_gap = max(10**-decimal_places, 0.1 * abs(correct_answer))

    for i in range(4):
        if i == correct_position:
            continue

        attempts = 0
        while attempts < 100:
            deviation = random.uniform(-deviation_range, deviation_range)
            distractor = correct_answer + deviation

            if isinstance(correct_answer, int):
                distractor = round(distractor)
            else:
                distractor = round(distractor, decimal_places)

            if (
                distractor not in options
                and distractor != correct_answer
                and all(
                    abs(distractor - opt) >= min_gap
                    for opt in options
                    if opt is not None
                )
            ):
                options[i] = distractor
                break

            attempts += 1

        if attempts == 100:
            options[i] = correct_answer + (i - correct_position) * min_gap

    return options, correct_option


def generate_interval_choices(correct_answer):
    total_range = abs(correct_answer) + 5
    min_value = correct_answer - total_range * random.uniform(0.5, 1)
    max_value = correct_answer + total_range * random.uniform(0.5, 1)

    split_points = [min_value, max_value]
    while len(split_points) < 5:
        split_point = random.uniform(min_value, max_value)
        if abs(split_point - correct_answer) > total_range * 0.05:
            split_points.append(split_point)

    split_points.sort()

    intervals = [(split_points[i], split_points[i + 1]) for i in range(4)]

    correct_interval_index = next(
        i
        for i, interval in enumerate(intervals)
        if interval[0] < correct_answer < interval[1]
    )
    correct_option = chr(65 + correct_interval_index)

    formatted_intervals = [f"[{interval[0]}, {interval[1]})" for interval in intervals]

    return formatted_intervals, correct_option


def generate_interval_choices_(correct_start, correct_end):
    correct_width = correct_end - correct_start
    correct_interval = f"[{correct_start}, {correct_end}]"

    correct_position = random.randint(0, 3)
    correct_option = chr(65 + correct_position)

    intervals = [None] * 4
    intervals[correct_position] = correct_interval

    for i in range(4):
        if i == correct_position:
            continue

        if i < correct_position:
            new_start = correct_start - (correct_position - i) * correct_width
        else:
            new_start = correct_end + (i - correct_position - 1) * correct_width

        new_end = new_start + correct_width
        intervals[i] = f"[{new_start}, {new_end}]"

    return intervals, correct_option


def generate_trend_quali(numbers):
    if len(numbers) < 2:
        return ["Stable", "Increasing", "Decreasing", "Fluctuating"], "Stable"

    start_value = numbers[0]
    end_value = numbers[-1]

    total_change_percent = (end_value - start_value) / start_value * 100

    max_value = max(numbers)
    min_value = min(numbers)
    max_fluctuation_percent = (max_value - min_value) / start_value * 100

    if abs(total_change_percent) <= 3:
        if max_fluctuation_percent > 3:
            trend = "Fluctuating, overall stable"
        else:
            trend = "Stable"
    elif abs(total_change_percent) <= 10:
        if total_change_percent > 0:
            trend = "Slight increase"
        else:
            trend = "Slight decrease"
    elif abs(total_change_percent) <= 20:
        if total_change_percent > 0:
            trend = "Moderate increase"
        else:
            trend = "Moderate decrease"
    else:
        if total_change_percent > 0:
            trend = "Significant increase"
        else:
            trend = "Significant decrease"

    is_steady = True
    prev_diff = None
    for i in range(1, len(numbers)):
        diff = numbers[i] - numbers[i - 1]
        if prev_diff is not None:
            if (diff > 0 and prev_diff < 0) or (diff < 0 and prev_diff > 0):
                is_steady = False
                break
        prev_diff = diff

    if trend not in ["Stable", "Fluctuating, overall stable"]:
        if is_steady:
            trend = "Steady " + trend
        else:
            trend = "Fluctuating, " + trend

    options = [
        "Stable",
        "Fluctuating, overall stable",
        "Steady slight increase",
        "Steady slight decrease",
        "Steady moderate increase",
        "Steady moderate decrease",
        "Steady significant increase",
        "Steady significant decrease",
        "Fluctuating, slight increase",
        "Fluctuating, slight decrease",
        "Fluctuating, moderate increase",
        "Fluctuating, moderate decrease",
        "Fluctuating, significant increase",
        "Fluctuating, significant decrease",
    ]

    if trend not in options:
        options.append(trend)
    random.shuffle(options)
    choices = [trend] + random.sample([opt for opt in options if opt != trend], 3)
    random.shuffle(choices)

    correct_option = chr(65 + choices.index(trend))

    return choices, correct_option


def generate_trend_quant(numbers):
    if len(numbers) < 2:
        return [
            "Stable",
            "Fluctuating, overall stable",
            "Steady increase by 1.0%",
            "Steady decrease by 1.0%",
        ], "Stable"

    start_value = numbers[0]
    end_value = numbers[-1]
    total_change_percent = (end_value - start_value) / start_value * 100
    max_fluctuation_percent = (max(numbers) - min(numbers)) / start_value * 100

    if abs(total_change_percent) < 3:
        if max_fluctuation_percent > 3:
            trend = "Fluctuating, overall stable"
        else:
            trend = "Stable"
    else:
        direction = "increase" if total_change_percent > 0 else "decrease"
        if max_fluctuation_percent > abs(total_change_percent) * 1.5:
            trend = f"{direction} by {abs(total_change_percent):.1f}%"
        else:
            trend = f"{direction} by {abs(total_change_percent):.1f}%"

        is_steady = True
        prev_diff = None
        for i in range(1, len(numbers)):
            diff = numbers[i] - numbers[i - 1]
            if prev_diff is not None:
                if (diff > 0 and prev_diff < 0) or (diff < 0 and prev_diff > 0):
                    is_steady = False
                    break
            prev_diff = diff

        if is_steady:
            trend = "Steady " + trend
        else:
            trend = "Fluctuating, " + trend

    distractor_options = [
        "Stable",
        "Fluctuating, overall stable",
        f"Steady increase by {random.uniform(0.1, max(abs(total_change_percent) * 2, 5)):.1f}%",
        f"Steady decrease by {random.uniform(0.1, max(abs(total_change_percent) * 2, 5)):.1f}%",
        f"Fluctuating, increase by {random.uniform(0.1, max(abs(total_change_percent) * 2, 5)):.1f}%",
        f"Fluctuating, decrease by {random.uniform(0.1, max(abs(total_change_percent) * 2, 5)):.1f}%",
    ]

    choices = [trend]
    while len(choices) < 4:
        choice = random.choice(distractor_options)
        if choice not in choices:
            choices.append(choice)

    random.shuffle(choices)

    correct_option = chr(65 + choices.index(trend))

    return choices, correct_option


def generate_simple_question(choices, correct_choice):
    random.shuffle(choices)
    correct_option = chr(65 + choices.index(correct_choice))
    return choices, correct_option