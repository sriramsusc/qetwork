"""A polarizing beam splitter"""

class PBS:
    # CITE diag-povm | imperfect-PBS terminal detection POVM is diagonal in the analyzer basis => exactly projective measurement + classical confusion matrix | Nielsen & Chuang (2010), §2.2.6 (POVM formalism)
    # CITE pbs-fiber | PM fused-fiber PBS @1550: PER >=22 dB, IL <=0.6 dB (defaults below) | commercial PM PBS specs (e.g. wdmquest / fiber-life 1x2 PBS 1550 nm)
    def __init__(self, t_p: float = 0.87, t_s = 5.5e-3, 
                 r_s: float = 0.87, r_p: float = 5.5e-3, 
                 band: tuple[float,float] = (1520, 1580)) -> None:
        for name, p in (("t_p", t_p), ("t_s", t_s), ("r_s", r_s), ("r_p", r_p)):
            if not 0 <= p <= 1:
                raise ValueError(f"{name} must be a probability in [0,1], got {p}")
        if t_p + r_p > 1:
            raise ValueError(f"p row exceeds unity: t_p+r_p= {t_p + r_p}")
        
        if t_s + r_s > 1:
            raise ValueError(f"p row exceeds unity: t_s+r_s= {t_s + r_s}")
        self.t_p = t_p
        self.r_p = r_p
        self.t_s = t_s
        self.r_s = r_s
        self.band = band

    @classmethod
    # CITE pbs-cube | Thorlabs broadband PBS 1200-1600 nm: Tp>90%, Rs>99.5%, ER_T>1000:1; reflected-port ER only ~20:1-100:1 (unspecified) | thorlabs.com broadband polarizing beamsplitter cubes (PBS254)
    def cube_raw(cls) -> "PBS":
        return cls(t_p = 0.90, t_s = 9e-4, r_p = 0.02, r_s = 0.995, band = (1200, 1600))

    @classmethod
    def cube_cleanup(cls) -> "PBS":
        return cls(t_p = 0.90, t_s = 9e-4, r_p = 1e-5, r_s = 0.85, band = (1200, 1600))
    
    @classmethod
    # CITE pbs-wollaston | calcite Wollaston prism: ER 100,000:1 both ports, 350 nm-2.3 um (uncoated at 1550) | Thorlabs Wollaston prisms (WP10)
    def wollaston(cls) -> "PBS":
        return cls(t_p = 0.90, t_s = 1e-5, r_p = 1e-5, r_s = 0.90, band = (350, 2300))

    def route(self, ideal_port: str, sample: float) ->str | None:
        """Route a collapsed outcome: transmitted | reflected | None(lost)"""

        if not 0 <= sample < 1:
            raise ValueError(f"sample must be in [0,1), got {sample}]")
        if ideal_port == "transmit":
            correct, wrong = ("transmitted", self.t_p), ("reflected", self.r_p)
        elif ideal_port == "reflect":
            correct, wrong = ("reflected", self.r_s), ("transmitted", self.t_s)
        else:
            raise ValueError(f"ideal port must be transmit or reflect, got {ideal_port}")
        if sample < correct[1]:
            return correct[0]
        if sample < correct[1] + wrong[1]:
            return wrong[0]

            