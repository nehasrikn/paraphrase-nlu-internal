elif isinstance(e, AbductiveNLIExample):
        return ANLI.format(
            obs1=e.obs1,
            obs2=e.obs2,
            hyp1=e.hyp1, 
            hyp2=e.hyp2
        )