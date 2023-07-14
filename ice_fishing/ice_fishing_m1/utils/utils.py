def catch_rate(last_catches: list, window_size: int = 50) -> float:
    """
    Estimate the catch rate based on the last catches
    """
    n_catches = sum(last_catches[:window_size])
    time_window = min(len(last_catches), window_size)
    return n_catches / time_window if time_window > 0 else 0.0
