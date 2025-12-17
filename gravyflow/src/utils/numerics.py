def ensure_even(number):
    """
    Ensures that a number is even by subtracting 1 if it is odd.
    """
    if number % 2 != 0:
        number -= 1
    return number
