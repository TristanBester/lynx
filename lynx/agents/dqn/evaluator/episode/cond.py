from lynx.types.types import EvalState


def create_cond_fn():
    def cond_fn(carry: EvalState) -> bool:
        """Check if the episode is done."""
        timestep = carry[2]
        is_not_done: bool = ~timestep.last()  # type: ignore
        return is_not_done

    return cond_fn
