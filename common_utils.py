# to keep values in range [-1,1]
def clamp(value):
    if value == 0.0:
        return value
    elif value > 0.0:
        return min(1.0, value)
    else:
        return max(-1.0, value)