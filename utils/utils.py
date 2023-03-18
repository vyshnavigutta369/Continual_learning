import numpy as np
import math
import os

def near_split(x, num_bins):
    if num_bins==0:
        return []
    quotient, remainder = divmod(x, num_bins)
    return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)

def scale_ratio(ratios: list) -> list:
    sum_ = sum(ratios)
    return [x/sum_ for x in ratios]

def ratio_breakdown_recursive(x: int, ratios: list) -> list:
    top_ratio = ratios[0]
    part = round(x*top_ratio)
    if x <= part:
        return [x]
    x -= part
    return [part] + ratio_breakdown_recursive(x, scale_ratio(ratios[1:]))

def ratio_breakdown(x: int, ratios: list) -> list:

    sorted_ratio = sorted(ratios, reverse=True)
    assert(round(sum(ratios)) == 1)
    sorted_result = ratio_breakdown_recursive(x, sorted_ratio)
    assert(sum(sorted_result) == x)
    # Now, we have to reverse the sorting and add missing zeros
    sorted_result += [0]*(len(ratios)-len(sorted_result))
    numbered_ratios = [(r, i) for i, r in enumerate(ratios)]
    sorted_numbered_ratios = sorted(numbered_ratios, reverse=True)
    combined = zip(sorted_numbered_ratios, sorted_result)
    combined_unsorted = sorted(combined, key=lambda x: x[0][1])
    unsorted_results = [x[1] for x in combined_unsorted]
    return unsorted_results

def getBack(var_grad_fn):
    # print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                # print(n[0])
                # print('Tensor with grad found:', tensor)
                if tensor.grad is None:
                    print ('Noooooo')
                # else:
                #     print ('Yesssss')
                # print(' - gradient:', tensor.grad)
                # print()
            except AttributeError as e:
                getBack(n[0])
