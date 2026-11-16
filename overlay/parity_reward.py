async def score(args, sample, **kwargs):
    # Alternating sample positions ensure a non-degenerate GRPO smoke update.
    return float(sample.index % 2)

