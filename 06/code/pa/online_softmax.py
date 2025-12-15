import math
import torch
import torch.nn.functional as F


def online_softmax(x):
    """
    Compute the softmax of a 1D tensor x in an online fashion (single pass)
    to improve numerical stability.
    """
    m = float('-inf')
    d = 0.0
    output = []

    for xi in x:
        if xi > m:
            # If we find a new max, we must rescale the previous sum
            # and the previously calculated exp values.
            # Correction factor: e^(old_max - new_max)
            correction = math.exp(m - xi)

            # Rescale the running denominator
            d = d * correction + 1.0  # +1.0 for the current xi (e^(xi-xi) = e^0 = 1)

            # Rescale all previously stored values to the new max
            output = [val * correction for val in output]

            # Update max
            m = xi

            # Append current value (scaled to itself, so it is 1.0)
            output.append(1.0)
        else:
            # Standard case: current value is smaller than current max
            val = math.exp(xi - m)
            d += val
            output.append(val)

    # Final normalization
    results = [val / d for val in output]
    return results


if __name__ == "__main__":
    x = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Run custom online softmax
    results = online_softmax(x)
    print("Online Softmax results:", results)

    # Compare with PyTorch
    # Note: torch.nn.functional.softmax is the functional interface
    softmax_torch = F.softmax(torch.tensor(x), dim=0).tolist()
    print("Torch Softmax results: ", softmax_torch)