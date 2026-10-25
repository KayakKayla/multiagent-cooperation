"""Local scenario extensions; the pinned JaxMARL checkout stays unchanged."""


def make_env(layout, **kwargs):
    if layout == "oc2_sushi_1_3":
        from .oc2_sushi_1_3.environment import SushiEnvironment
        return SushiEnvironment(**kwargs)
    import jaxmarl
    return jaxmarl.make("overcooked_v2", layout=layout, **kwargs)
