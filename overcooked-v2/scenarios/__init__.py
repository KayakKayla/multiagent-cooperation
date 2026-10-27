"""Local scenario extensions; the pinned JaxMARL checkout stays unchanged."""


def make_env(layout, env_name="overcooked_v2", **kwargs):
    if env_name == "overcooked_v2" and layout == "oc2_sushi_1_3":
        from .oc2_sushi_1_3.environment import Rules, SushiEnvironment
        if isinstance(kwargs.get("rules"), dict):
            kwargs["rules"] = Rules(**kwargs["rules"])
        return SushiEnvironment(**kwargs)
    import jaxmarl
    return jaxmarl.make(env_name, layout=layout, **kwargs)


def wrap_training(env):
    if env.name == "oc2_sushi_1_3":
        from .oc2_sushi_1_3.training import SushiLogWrapper
        return SushiLogWrapper(env)
    from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
    return OvercookedV2LogWrapper(env, replace_info=False)
