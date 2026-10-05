import os
import pandas as pd
import numpy as np

# SPARC data files (download from http://astroweb.case.edu/SPARC/).
# Override the location with the SPARC_DATA environment variable, or place the
# two files in a ``data/`` directory at the repository root.
_DATA = os.environ.get("SPARC_DATA",
                       os.path.join(os.path.dirname(__file__), "..", "data"))
T1 = os.path.join(_DATA, "SPARC_Lelli2016c_mrt.txt")
T2 = os.path.join(_DATA, "SPARC_MassModels_Lelli2016c_mrt.txt")


def load_table1():
    names = ["Galaxy", "T", "D", "e_D", "f_D", "Inc", "e_Inc", "L36", "e_L36",
             "Reff", "SBeff", "Rdisk", "SBdisk", "MHI", "RHI", "Vflat",
             "e_Vflat", "Q", "Ref"]
    rows = []
    with open(T1) as f:
        for line in f.readlines()[98:]:
            p = line.split()
            if len(p) == 19:
                rows.append(p)
            elif len(p) == 18:          # missing reference string
                rows.append(p + [""])
            elif p:
                raise ValueError(f"bad Table1 line ({len(p)} fields): {line!r}")
    df = pd.DataFrame(rows, columns=names)
    for c in names[1:-1]:
        df[c] = pd.to_numeric(df[c])
    return df.set_index("Galaxy")


def load_table2():
    cols = [(0, 11), (12, 18), (19, 25), (26, 32), (33, 38), (39, 45),
            (46, 52), (53, 59), (60, 67), (68, 76)]
    names = ["Galaxy", "D", "R", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul",
             "SBdisk", "SBbul"]
    df = pd.read_fwf(T2, colspecs=cols, names=names, skiprows=25)
    df["Galaxy"] = df["Galaxy"].str.strip()
    return df


TYPES = {0: "S0", 1: "Sa", 2: "Sab", 3: "Sb", 4: "Sbc", 5: "Sc", 6: "Scd",
         7: "Sd", 8: "Sdm", 9: "Sm", 10: "Im", 11: "BCD"}
